#!/usr/bin/env python3
"""textbook-mcp — MCP server for a local medical textbook corpus (stdio).

Environment variables:
    TEXTBOOK_DB       path to corpus.sqlite (optional, default ./corpus.sqlite)
    TEXTBOOK_DENSE    0 = BM25 only (no model is loaded; fast and light)
    TEXTBOOK_RERANK   0 = cross-encoder disabled
    TEXTBOOK_WARMUP   1 = models + matrix are warmed up AT STARTUP (default: lazy)
    TEXTBOOK_IDLE_UNLOAD  if no request arrives for N seconds, the models + matrix are
                      dropped from memory (default 60; 0 = off). The process STAYS ALIVE
                      and the stdio connection is kept; the cost is a reload on the next query.

Design: the server ONLY searches and retrieves. Synthesis/interpretation is done by the
calling model (Claude) — the same division of labor as the DynaMed MCP.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

# In stdio MCP, stdout is for the protocol ONLY. Pin library logs to stderr
# and run HF offline (the models are already local; otherwise every startup hits the network).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.basicConfig(level=logging.WARNING, stream=__import__("sys").stderr)
for _n in ("sentence_transformers", "transformers", "httpx", "huggingface_hub", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import threading

from mcp.server.mcpserver import MCPServer, Image
from mcp.server.mcpserver.exceptions import ToolError

from . import search as S

# The default path is relative to the package, not the working directory: the server
# finds the same corpus whatever cwd it is started from.
DB_PATH = os.environ.get("TEXTBOOK_DB") or str(
    Path(__file__).resolve().parent.parent / "corpus.sqlite")
USE_DENSE = os.environ.get("TEXTBOOK_DENSE", "1") != "0"
USE_RERANK = os.environ.get("TEXTBOOK_RERANK", "1") != "0"
# Warmup is now OPT-IN. stdio MCP servers are spawned as a separate process per
# session; unconditional warmup allocated matrix (1.8 GB) + bge-m3 + reranker
# = ~8 GB in every session, even one that never used textbook. With 7 concurrent
# sessions this reached 56 GB and buried the 18 GB machine in 42 GB of swap (measured).
# The lazy singletons already load on first call; the cost is ~8 s on the first search.
USE_WARMUP = os.environ.get("TEXTBOOK_WARMUP", "0") == "1"
# Idle unload: the counterpart of warmup. Warmup allocated memory in sessions that
# NEVER used it; this prevents memory from being held indefinitely in sessions that DID.
# Measured 2026-09-22, LIVE server (Apple MPS): RSS 2502 MB after a search;
# 926 MB 60 s after unloading, and STABLE there (no drift over 2 min).
# Gain per session ~1.6 GB. The remaining 926 MB is NOT the models -- corpus_info
# shows all three "su_an_bellekte" flags as false; it is the Python/torch allocator
# not returning to macOS the arena left over from the 982,878-row matrix. A session
# that never searched stays at ~76 MB (lazy loading). With 6 concurrent sessions
# swap had reached 12 GB. Cost: a cold reload on the next search
# (measured 11.9-16.3 s; the first load is 391+393 weight shards).
IDLE_UNLOAD = max(0, int(os.environ.get("TEXTBOOK_IDLE_UNLOAD", "60")))
_last_activity = time.monotonic()
_activity_lock = threading.Lock()


_inflight = 0


def _tracked(fn):
    """Count the tool call as 'in flight' and reset the clock when it FINISHES.

    Looking only at entry was not enough: the first search, which loads the models, takes
    20 s, and because activity was not refreshed during that time the watchdog unloaded in
    the MIDDLE of the work (measured, reproduced with a 5 s threshold). While the counter is
    above zero the watchdog does not touch anything; the timer starts when the call ends --
    exactly the definition "N s after the query finished". Nested calls are safe
    (find_in_book -> search).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _inflight, _last_activity
        with _activity_lock:
            _inflight += 1
        try:
            return fn(*args, **kwargs)
        finally:
            with _activity_lock:
                _inflight -= 1
                _last_activity = time.monotonic()
    return wrapper

mcp = MCPServer("textbook", instructions=__doc__)
_con: sqlite3.Connection | None = None
_db_lock = threading.Lock()


def db() -> sqlite3.Connection:
    global _con
    if _con is None:
        with _db_lock:
            if _con is None:
                p = Path(DB_PATH).expanduser()
                if not p.exists():
                    raise ToolError(
                        f"Corpus not found: {p}. Run `python -m tbmcp.ingest` first.")
                # READ-ONLY: the server never writes; it must not race for locks with
                # an embed job writing to the same file concurrently.
                _con = sqlite3.connect(f"file:{p}?mode=ro", uri=True,
                                       check_same_thread=False)
                # mmap: OFF BY DEFAULT -- measured, no gain.
                # 2026-09-22, 982,878-row matrix, 4 repetitions each in separate
                # processes: on mean 2.85 s (median 2.08), off mean 2.61 s
                # (median 2.32). The difference is within the noise; each series has
                # one outlier (5.33 / 3.86 s). The acceptance criterion was >=10%.
                # CAUTION: only the WARM state was measured. A COLD start (the only
                # case where mmap is expected to help) could not be measured: `purge`
                # needs root. To measure cold, after `sudo purge`:
                #   TEXTBOOK_MMAP=2147483648 .venv/bin/python -c "..."
                mmap_bytes = int(os.environ.get("TEXTBOOK_MMAP", 0))
                if mmap_bytes:
                    _con.execute(f"PRAGMA mmap_size={mmap_bytes}")
                # In WAL mode a reader is not blocked by the writer; this is only cheap
                # insurance for checkpoint/recovery moments. No gain is claimed.
                _con.execute("PRAGMA busy_timeout=5000")
    return _con


def _book_id(slug: str) -> tuple[int, str, int]:
    row = db().execute(
        "SELECT id,title,n_pages FROM books WHERE slug=?", (slug,)).fetchone()
    if not row:
        known = [r[0] for r in db().execute("SELECT slug FROM books LIMIT 10")]
        raise ToolError(f"Unknown book: {slug!r}. Examples: {known}")
    return row


# ----------------------------------------------------------------------- tools
@mcp.tool()
@_tracked
def textbook_list_books() -> str:
    """List all textbooks in the corpus (slug, title, page count).

    The `book` parameter of the other tools expects the `slug` value shown here.
    Use it to see which sources the corpus contains before answering a question.
    """
    rows = db().execute(
        "SELECT b.slug,b.title,b.n_pages,COUNT(c.id),b.scanned,b.collection,b.lang"
        " FROM books b LEFT JOIN chunks c ON c.book_id=b.id"
        " GROUP BY b.id ORDER BY b.collection, b.title").fetchall()
    if not rows:
        return "The corpus is empty."
    out = [f"{len(rows)} books:\n"]
    cur = None
    for slug, title, pages, n, scanned, coll, lang in rows:
        if coll != cur:
            cur = coll
            kac = sum(1 for r in rows if r[5] == coll)
            out.append(f"\n== collection: {coll} ({lang}) — {kac} sources ==")
        flag = "  [SCANNED/no OCR]" if scanned else ""
        out.append(f"- {slug}\n    {title}\n    {pages} pages, {n} chunks{flag}")
    return "\n".join(out)


@mcp.tool()
@_tracked
def textbook_search(query: str, book: str = "", top_k: int = 8) -> str:
    """Hybrid search across the textbooks (BM25 + semantic + reranking).

    This is the first step for clinical questions. Write the query in the language of
    the books (in most corpora, English medical terms). A query in another language
    only works while the semantic layer is on — first check the dense_acik value with
    textbook_corpus_info; if it is false, the query MUST be in the language of the books.

    SCOPE: only the 'textbook' collection. For lecture notes (the 'notes' collection)
    use textbook_notes_search — this tool does NOT return them.
    The lock is deliberate: measured, for a query in the notes' language, note chunks
    entered the 30-candidate pool through BM25 and could push a high-scoring textbook
    chunk out before it ever reached the reranker. When the two collections are
    searched separately, both get full recall.

    Args:
        query: Search phrase, e.g. "diabetic ketoacidosis insulin infusion rate"
        book:  Slug to restrict the search to a single book (empty = the whole textbook corpus)
        top_k: Number of chunks to return (default 8)

    Returns: text chunks, each with a [book s.N] citation. Carry the citation over
    verbatim in your answer; open the page with textbook_get_page for full context.
    """
    hits = S.search(db(), query, top_k=max(1, min(top_k, 25)),
                    book=book or None, dense=USE_DENSE, use_rerank=USE_RERANK,
                    collection="textbook")
    return _fmt_hits(hits, query,
                     "Not found in the textbooks; for the lecture notes "
                     "try textbook_notes_search.")


def _fmt_hits(hits, query: str, bos_ipucu: str = "") -> str:
    """Format search results as cited text (shared by textbook_search and textbook_notes_search)."""
    if not hits:
        return (f"No results in the corpus for '{query}'. "
                + (bos_ipucu or "Change the term and try again, or "
                                "see the coverage with textbook_list_books."))
    out = []
    for i, h in enumerate(hits, 1):
        head = f"{i}. {h.cite()}  score={h.score:.3f}"
        if h.section:
            head += f"  |  {h.section[:80]}"
        out.append(f"{head}\n{h.text}")
    return "\n\n---\n\n".join(out)


@mcp.tool()
@_tracked
def textbook_notes_search(query: str, book: str = "", top_k: int = 8) -> str:
    """Search the lecture notes (the 'notes' collection).

    Use THIS for a question asked in the language of the lecture notes; for the
    textbooks use textbook_search. The two are SEPARATE sources and do not return
    each other's results; for a thorough answer you can call both.

    Why a separate tool: measured — because the textbook corpus is tens of times
    larger and same-language matches score higher than cross-language ones, note
    chunks searched in a single pool could not surface even for queries in their
    own language.

    Args:
        query: Phrase in the language of the notes, e.g. "diyabetik ketoasidoz insulin dozu"
        book:  Slug to restrict the search to a single notes file (see textbook_list_books)
        top_k: Number of chunks to return (default 8)

    Returns: text chunks with a [note s.N] citation. Tables can be garbled in the
    text layer — if the numbers look scrambled column by column, call
    textbook_page_image for the same page.
    """
    hits = S.search(db(), query, top_k=max(1, min(top_k, 25)),
                    book=book or None, dense=USE_DENSE, use_rerank=USE_RERANK,
                    collection="notes")
    return _fmt_hits(hits, query,
                     "Not found in the lecture notes; for the textbooks "
                     "try textbook_search.")


@mcp.tool()
@_tracked
def textbook_get_page(book: str, page: int, window: int = 1) -> str:
    """Return a specific page of a book (and its neighbors) as full text.

    Use it after textbook_search returns a chunk, to complete the truncated
    context. window=1 also returns the previous and the next page.

    Args:
        book:   book slug
        page:   PDF page number (the s.N value in search results)
        window: how many neighboring pages (0-5)
    """
    bid, title, n_pages = _book_id(book)
    if not 1 <= page <= n_pages:
        raise ToolError(f"{book} has {n_pages} pages; there is no s.{page}.")
    window = max(0, min(window, 5))
    lo, hi = max(1, page - window), min(n_pages, page + window)
    rows = db().execute(
        "SELECT page,text FROM pages WHERE book_id=? AND page BETWEEN ? AND ?"
        " ORDER BY page", (bid, lo, hi)).fetchall()
    if not rows:
        return f"{book} s.{page} is empty or missing (the book has {n_pages} pages)."
    parts = [f"# {title}"]
    total = 0
    for p, txt in rows:
        seg = f"\n----- [{book} s.{p}] -----\n{txt or '(empty page)'}"
        total += len(seg)
        parts.append(seg)
        if total > 60_000:        # ~15k token cap: 11 huge pages must not bloat the context
            parts.append(f"\n[truncated — reduce window to see the remaining pages]")
            break
    return "\n".join(parts)


@mcp.tool()
@_tracked
def textbook_find_in_book(book: str, query: str, top_k: int = 10) -> str:
    """Search within a single book. Use it to learn what one source says about a
    particular topic, or to compare books.
    """
    return textbook_search(query=query, book=book, top_k=top_k)


@mcp.tool()
@_tracked
def textbook_toc(book: str, depth: int = 2, contains: str = "") -> str:
    """Return a book's table of contents tree (from the PDF bookmarks).

    Args:
        book:     book slug
        depth:    how many levels deep to go (1-6)
        contains: only headings containing this phrase (empty = all)
    """
    bid, title, _ = _book_id(book)
    sql = "SELECT level,title,page FROM toc WHERE book_id=? AND level<=?"
    params: list = [bid, max(1, min(depth, 6))]
    if contains:
        sql += " AND title LIKE ?"
        params.append(f"%{contains}%")
    sql += " ORDER BY page,level LIMIT 800"
    rows = db().execute(sql, params).fetchall()
    if not rows:
        return f"{book}: no PDF bookmarks. Search with textbook_search."
    return f"# {title}\n" + "\n".join(
        f"{'  ' * (lvl - 1)}{t.strip()}  ....  s.{pg}" for lvl, t, pg in rows)


@mcp.tool()
@_tracked
def textbook_page_image(book: str, page: int, dpi: int = 150) -> Image:
    """Return a page as an IMAGE. Use it for tables, flowcharts, algorithms and figures
    — these get garbled when converted to plain text.

    Args:
        book: book slug
        page: PDF page number
        dpi:  resolution (72-200)
    """
    import pymupdf
    row = db().execute("SELECT path, n_pages, sha256 FROM books WHERE slug=?",
                       (book,)).fetchone()
    if not row:
        raise ToolError(f"Unknown book: {book!r}")
    path, n_pages, sha = row
    if not os.path.isabs(path):
        # A relative path would resolve against the server's cwd, and the cwd is deliberately
        # unspecified (see the DB_PATH note) — stop rather than silently open the wrong file.
        raise ToolError(f"{book}: books.path is relative ({path!r}); it must be made absolute.")
    try:
        doc = pymupdf.open(path)
    except Exception:
        raise ToolError(f"Could not open the PDF (it may have been moved): {path}")
    with doc:
        # If the path stays the same but the file is replaced with ANOTHER edition, page
        # numbers shift and the tool would silently show the wrong page. The page count is a
        # free identity check (the document is already open); a different edition almost
        # always has a different page count. For exact verification there is books.sha256.
        if n_pages and doc.page_count != n_pages:
            raise ToolError(
                f"{book}: the PDF has {doc.page_count} pages, the corpus expects {n_pages} — "
                f"the file may have changed (a different edition?). Verify: "
                f"shasum -a 256 {path!r}  ->  expected {sha or '(not recorded)'}")
        if not 1 <= page <= doc.page_count:
            raise ToolError(f"{book} has {doc.page_count} pages; there is no s.{page}.")
        # Cap the long edge at 1568 px: the API downscales anything above 1568 anyway,
        # and the excess is bandwidth that never reaches the model. A large-format page at
        # 200 dpi produced a 6+ MB PNG; the cap + JPEG fallback brings the worst case under ~1 MB.
        rect = doc[page - 1].rect
        zoom = min(max(72, min(dpi, 200)) / 72,
                   1568 / max(rect.width, rect.height))
        pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        png = pix.tobytes("png")
        if len(png) > 800_000:
            return Image(data=pix.tobytes("jpg", jpg_quality=80), format="jpeg")
        return Image(data=png, format="png")


@mcp.tool()
@_tracked
def textbook_corpus_info() -> str:
    """Corpus status: book/chunk counts, whether the vector layer is ready, which model."""
    c = db()
    n_books, n_chunks, n_pages, n_vec = (
        c.execute("SELECT COUNT(*) FROM books").fetchone()[0],
        c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        c.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
        c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    meta = dict(c.execute("SELECT k,v FROM meta").fetchall())
    koll = {
        r[0]: {"kitap": r[1], "sayfa": r[2], "parca": r[3], "dil": r[4],
               "arac": ("textbook_notes_search" if r[0] == "notes"
                        else "textbook_search")}
        for r in c.execute(
            "SELECT b.collection,COUNT(DISTINCT b.id),SUM(b.n_pages),"
            "(SELECT COUNT(*) FROM chunks k JOIN books b2 ON b2.id=k.book_id"
            "  WHERE b2.collection=b.collection),MIN(b.lang)"
            " FROM books b GROUP BY b.collection")}
    return json.dumps({
        "db": str(Path(DB_PATH).expanduser()),
        "kitap": n_books, "sayfa": n_pages, "parca": n_chunks,
        "koleksiyonlar": koll,
        "vektor": n_vec,
        "vektor_kapsami": f"{100*n_vec/n_chunks:.1f}%" if n_chunks else "0%",
        "dense_acik": USE_DENSE, "rerank_acik": USE_RERANK,
        "bosta_bosaltma_sn": IDLE_UNLOAD,
        "ucustaki_istek": _inflight,
        "su_an_bellekte": S.loaded(),
        **meta,
    }, ensure_ascii=False, indent=2)


def _warmup() -> None:
    """Warm up the models and the matrix IN THE BACKGROUND after the handshake.

    Previously the first textbook_search call carried ~8 s of model loading
    (including Metal shader compilation). The locks (search._model_lock/_mat_lock) prevent
    a double load with a first query arriving at the same time. The warmup includes a tiny
    encode + rerank pair: merely constructing the object does not pay for shader compilation.
    """
    try:
        con = db()
        if USE_DENSE:
            from .embed import encode, get_device
            S._fresh_matrix(con)
            encode(["warmup"], get_device())
        if USE_RERANK:
            S.rerank("warmup", [S.Hit(0, "w", "w", 1, 1, "", "warmup text")], 1)
    except Exception as e:                      # a warmup error is not fatal
        logging.warning("warmup skipped: %s", e)


def _idle_watchdog() -> None:
    """Drop the models/matrix if no request arrives for IDLE_UNLOAD s, then go back to waiting.

    The process does NOT die: the stdio pipe stays open and the client does not have to
    reconnect. The wake-up interval follows the remaining time; after an unload it quietly
    loops every 30 s (no cost when unload() returns empty-handed).
    """
    while True:
        with _activity_lock:
            idle = time.monotonic() - _last_activity
            busy = _inflight
        if busy == 0 and idle >= IDLE_UNLOAD:
            try:
                if S.unload():
                    logging.warning("idle %.0f s -> models dropped from memory", idle)
            except Exception as e:                  # an unload error is not fatal
                logging.warning("idle unload failed: %s", e)
            time.sleep(min(IDLE_UNLOAD, 30))
        else:
            time.sleep(max(1.0, IDLE_UNLOAD - idle))


def main() -> None:
    if USE_WARMUP:
        threading.Thread(target=_warmup, daemon=True).start()
    if IDLE_UNLOAD > 0:
        threading.Thread(target=_idle_watchdog, daemon=True).start()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
