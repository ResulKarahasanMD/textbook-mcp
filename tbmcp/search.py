#!/usr/bin/env python3
"""Hybrid search: BM25 (FTS5) ∪ dense (bge-m3) -> RRF -> cross-encoder rerank.

Also works as a CLI (the example query is Turkish):
    python -m tbmcp.search --db corpus.sqlite "diyabetik ketoasidoz tedavisi"
"""
from __future__ import annotations

import argparse
import gc
import re
import sqlite3
import sys
import threading
from dataclasses import dataclass, field

import numpy as np

RERANKER = "BAAI/bge-reranker-v2-m3"
RRF_K = 60  # default from Cormack et al. (2009)

# (ids, mat, max_chunk_id) — max_chunk_id is for the freshness probe (see _fresh_matrix)
_matrix: tuple[np.ndarray, np.ndarray, int] | None = None
# collection -> bool mask; invalidated by the same max_chunk_id probe as _matrix
_coll_masks: dict[str, np.ndarray] = {}
_coll_masks_max: int = -1
# For a filtered search (single book / small collection) only those rows are loaded into
# memory instead of the whole matrix (see _subset_matrix). (book, collection) -> (ids, mat,
# max_chunk_id); the _SUBSET_KEEP most recently used ones are kept.
SUBSET_MAX_ROWS = 200_000
_SUBSET_KEEP = 2
SCORE_BLOCK_ROWS = 2048                 # fp32 working buffer for scoring (see dense_search)
_subsets: dict[tuple[str | None, str | None], tuple[np.ndarray, np.ndarray, int]] = {}
# filters exceeding SUBSET_MAX_ROWS -> max_chunk_id at decision time (to avoid counting on every query)
_subset_big: dict[tuple[str | None, str | None], int] = {}
_reranker = None
# Without locks the lazy singletons raced under parallel MCP tool calls; moreover,
# torch/MPS models are not documented as safe for concurrent multi-threaded
# inference — _model_lock serializes both init and the encode/predict calls.
_mat_lock = threading.Lock()
_model_lock = threading.Lock()


def loaded() -> dict[str, bool]:
    """What is in memory right now. corpus_info uses it to verify idle unloading."""
    from . import embed
    return {"model": embed._model is not None,
            "reranker": _reranker is not None,
            "matris": _matrix is not None or bool(_subsets)}


def unload() -> bool:
    """Drop the model, the reranker and the vector matrix from memory.

    Lock order is the SAME as in dense_search: first _model_lock (embed's inside it),
    then _mat_lock. The two are never held nested -> no deadlock.
    A search in progress holds its own local reference to ids/mat, so unloading
    does not break it; the memory is freed when that work finishes.
    Returns: True if something was actually unloaded.
    """
    global _matrix, _coll_masks, _coll_masks_max, _reranker
    freed = False
    with _model_lock:
        if _reranker is not None:
            _reranker = None
            freed = True
        from .embed import unload_model
        if unload_model():
            freed = True
    with _mat_lock:
        if _matrix is not None:
            _matrix = None
            freed = True
        if _coll_masks:
            _coll_masks = {}
            _coll_masks_max = -1
            freed = True
        if _subsets:
            _subsets.clear()
            freed = True
        _subset_big.clear()
    if freed:
        gc.collect()
        # The torch cache is kept separately; gc does not see it. Don't import torch if it is not
        # loaded: importing it just to unload grew the process by ~120 MB (measured, in a
        # process that used only subsets).
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    return freed


@dataclass
class Hit:
    chunk_id: int
    book: str
    title: str
    page_start: int
    page_end: int
    section: str
    text: str
    n_pages: int = 0          # total pages of the book; for index/back-matter detection
    score: float = 0.0
    why: list[str] = field(default_factory=list)  # which layer found it

    def cite(self) -> str:
        pg = (f"s.{self.page_start}" if self.page_start == self.page_end
              else f"s.{self.page_start}-{self.page_end}")
        return f"[{self.book} {pg}]"


# ------------------------------------------------------------------------ BM25
# FTS5 unicode61 counts letters+digits as ONE token ("hba1c", "b12", "cd4", "t3").
# Query tokenization must use the same boundary: the old version split letter/digit
# runs and produced HbA1c -> '"HbA" OR "1"' (measured: 0 real matches,
# 191k chunks of noise). Branches: a real decimal ("7.35"), a letter+digit run, a bare number.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)+|[^\W_]{2,}|\d+", re.UNICODE)


def fts_query(q: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression (an OR of the terms)."""
    terms = _TOKEN.findall(q)
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms[:32])


def bm25_search(con: sqlite3.Connection, query: str, limit: int,
                book: str | None = None,
                collection: str | None = None) -> list[int]:
    m = fts_query(query)
    if not m:
        return []
    sql = ("SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
           "JOIN books b ON b.id = c.book_id WHERE f.chunks_fts MATCH ?")
    params: list = [m]
    if book:
        sql += " AND b.slug = ?"
        params.append(book)
    if collection:
        # SQL pre-filter: the filter makes BM25 cheaper because the candidate pool shrinks
        # (measured: whole corpus 95 ms, single book 61 ms)
        sql += " AND b.collection = ?"
        params.append(collection)
    sql += " ORDER BY bm25(chunks_fts, 1.0, 0.4) LIMIT ?"
    params.append(limit)
    return [r[0] for r in con.execute(sql, params)]


# ----------------------------------------------------------------------- dense
def _fresh_matrix(con: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the vector matrix; while an embed job runs, append new vectors INCREMENTALLY.

    Probe: SELECT MAX(chunk_id) over the PK takes ~0.1 ms (COUNT(*) measured 90 ms).
    embed.py only INSERTs new chunk_ids, so existing rows never change;
    incremental fetching is therefore correct. Old behavior: the cache was never refreshed,
    and the server stayed frozen at an X% snapshot of the embed job.
    """
    global _matrix
    row = con.execute("SELECT MAX(chunk_id) FROM embeddings").fetchone()
    cur_max = int(row[0] or 0)
    if _matrix is not None and _matrix[2] == cur_max:
        return _matrix
    with _mat_lock:
        from .embed import load_matrix
        if _matrix is None or len(_matrix[0]) == 0:
            ids, mat = load_matrix(con)
            _matrix = (ids, mat, cur_max)
        elif _matrix[2] != cur_max:
            ids0, mat0, last = _matrix
            rows = con.execute(
                "SELECT chunk_id, vec FROM embeddings WHERE chunk_id > ?"
                " ORDER BY chunk_id", (last,)).fetchall()
            if rows:
                new_ids = np.array([r[0] for r in rows], dtype=ids0.dtype)
                new_mat = np.frombuffer(b"".join(r[1] for r in rows),
                                        dtype=np.float16).reshape(len(rows), mat0.shape[1])
                _matrix = (np.concatenate([ids0, new_ids]),
                           np.vstack([mat0, new_mat]), cur_max)
            else:
                _matrix = (ids0, mat0, cur_max)
    return _matrix


def _id_mask(con: sqlite3.Connection, ids: np.ndarray, sql: str,
             params: tuple) -> np.ndarray:
    """Bool mask over ids that is True for the chunk ids returned by the SQL.

    The old version was `np.fromiter((i in allowed for i in ids), ...)`: measured 65.8 ms
    on 983k rows, a pure Python loop. np.isin gives the same result in 2.5 ms
    (26x), and that cost was paid on every textbook_find_in_book call.
    """
    allowed = np.fromiter((r[0] for r in con.execute(sql, params)),
                          dtype=ids.dtype)
    return np.isin(ids, allowed)


def _collection_mask(con: sqlite3.Connection, ids: np.ndarray, cur_max: int,
                     collection: str) -> np.ndarray:
    """Return the collection mask from the cache; rebuild it if the matrix was refreshed."""
    global _coll_masks, _coll_masks_max
    with _mat_lock:
        if _coll_masks_max != cur_max:
            _coll_masks, _coll_masks_max = {}, cur_max
        m = _coll_masks.get(collection)
        if m is None or len(m) != len(ids):
            m = _id_mask(con, ids,
                         "SELECT c.id FROM chunks c JOIN books b ON b.id=c.book_id"
                         " WHERE b.collection=?", (collection,))
            _coll_masks[collection] = m
    return m


def _subset_matrix(con: sqlite3.Connection, book: str | None,
                   collection: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the vectors of the rows matching the filter: (ids, mat). None if the filter is large.

    The full matrix takes 2.5 GB at 1.24 M rows; a search restricted to one book needs at
    most 51,648 of those rows (106 MB). If the filter exceeds SUBSET_MAX_ROWS (the textbook
    collection = 98% of the corpus), None is returned and the caller falls back to the full
    matrix. Freshness uses the same probe as _fresh_matrix: if max_chunk_id changed, the
    subset is fetched again.
    """
    from .embed import empty_f16
    key = (book, collection)
    cur_max = int(con.execute("SELECT MAX(chunk_id) FROM embeddings").fetchone()[0] or 0)
    where = " AND ".join(w for w, v in (("b.slug=?", book), ("b.collection=?", collection)) if v)
    params = tuple(v for v in (book, collection) if v)
    with _mat_lock:
        hit = _subsets.get(key)
        if hit is not None and hit[2] == cur_max:
            _subsets[key] = _subsets.pop(key)           # most recently used goes last
            return hit[0], hit[1]
        if _subset_big.get(key) == cur_max:
            return None
        # Upper bound from the chunks_book index; counted without joining embeddings.
        n_max, = con.execute("SELECT COUNT(*) FROM chunks c JOIN books b ON b.id=c.book_id"
                             " WHERE " + where, params).fetchone()
        if n_max > SUBSET_MAX_ROWS:
            _subset_big[key] = cur_max
            return None
        cur = con.execute("SELECT e.chunk_id, e.vec FROM embeddings e"
                          " JOIN chunks c ON c.id=e.chunk_id JOIN books b ON b.id=c.book_id"
                          " WHERE " + where + " ORDER BY e.chunk_id", params)
        rows = cur.fetchmany(8192)
        dim = len(rows[0][1]) // 2 if rows else 0       # float16 -> 2 bytes
        ids = np.empty(n_max, dtype=np.int64)
        mat = empty_f16(n_max, dim)                     # mmap: returns to the OS when dropped
        i = 0
        while rows:
            k = min(len(rows), n_max - i)               # overflow if the embed job adds rows meanwhile
            if k <= 0:
                break
            ids[i:i + k] = [r[0] for r in rows[:k]]
            mat[i:i + k] = np.frombuffer(
                b"".join(r[1] for r in rows[:k]), dtype=np.float16).reshape(k, dim)
            i += k
            rows = cur.fetchmany(8192)
        cur.close()
        ids, mat = ids[:i], mat[:i]
        _subsets.pop(key, None)
        _subsets[key] = (ids, mat, cur_max)
        while len(_subsets) > _SUBSET_KEEP:
            _subsets.pop(next(iter(_subsets)))
    return ids, mat


def dense_search(con: sqlite3.Connection, query: str, limit: int,
                 book: str | None = None,
                 collection: str | None = None) -> list[int]:
    from .embed import get_device, encode

    # If the full matrix is already in memory (an unfiltered search ran), mask it; otherwise
    # fetch only the rows needed for the filtered search.
    sub = None
    if (book or collection) and _matrix is None:
        sub = _subset_matrix(con, book, collection)
    if sub is not None:
        ids, mat = sub
    else:
        ids, mat, cur_max = _fresh_matrix(con)
    if len(ids) == 0:
        return []

    with _model_lock:
        qv = encode([query], get_device())[0].astype(np.float32)
    # Copy the fp16 matrix block by block into ONE fp32 buffer and multiply: a full-matrix
    # astype allocated a ~2 GB temporary per query (measured: 679 ms -> 75 ms, results
    # bit-for-bit identical). 64k-row `astype` blocks also left a 268 MB temporary each,
    # and freed large blocks do not drop out of the footprint on this machine
    # (measured: a deleted 762 MB stayed in MALLOC_LARGE). The fixed 2048-row buffer is 8 MB;
    # 31.4 -> 21.6 ms at 300k rows, result bit-for-bit identical. Vectors are normalized,
    # so dot = cosine.
    scores = np.empty(len(ids), np.float32)
    buf = np.empty((min(SCORE_BLOCK_ROWS, len(ids)), mat.shape[1]), np.float32)
    for i in range(0, len(ids), SCORE_BLOCK_ROWS):
        k = min(SCORE_BLOCK_ROWS, len(ids) - i)
        np.copyto(buf[:k], mat[i:i + k])
        np.dot(buf[:k], qv, out=scores[i:i + k])

    mask = None
    if sub is None and collection:
        mask = _collection_mask(con, ids, cur_max, collection)
    if sub is None and book:
        m = _id_mask(con, ids,
                     "SELECT c.id FROM chunks c JOIN books b ON b.id=c.book_id"
                     " WHERE b.slug=?", (book,))
        mask = m if mask is None else (mask & m)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    n = min(limit, len(ids))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    return [int(ids[i]) for i in top if np.isfinite(scores[i])]


# --------------------------------------------------- reference filtering
# End-of-chapter reference lists in textbooks mislead BM25 because they are dense with
# query terms (measured: "tumor lysis syndrome rasburicase" -> the first two hits were a
# Nathan & Oski citation list). They are citation dumps, not clinical content, so they
# are demoted rather than dropped: if there is no other candidate they are still returned.
_CITE_PAT = re.compile(
    r"\bet\s+al\b"                      # "Annemans L, Moeremans K, et al"
    r"|\b\d{1,4}:\d{1,4}\s*[-–]\s*\d{1,4}\b"   # "14:1203-1210" volume:page
    r"|\bdoi:"
    r"|\b(?:19|20)\d{2}\s*[.;]"         # "2003." year at the end
    r"|^\s*\d{1,4}\.\s+[A-Z]",          # "108. Annemans" numbered list
    re.I | re.M)


def is_reference_dump(text: str, min_hits: int = 4) -> bool:
    """Is the chunk a reference/citation list? Looks at density; a single citation won't trip it."""
    hits = len(_CITE_PAT.findall(text))
    if hits < min_hits:
        return False
    # 1.5+ citation markers per 400 characters -> a citation dump, not content
    return hits / max(1.0, len(text) / 400) >= 1.5


# Back-of-book indexes mislead BM25 the same way: query terms from start to finish,
# page numbers sprinkled in between, zero prose. They don't fit the reference pattern,
# so _CITE_PAT missed them (measured: in a laboratory medicine book the FIRST hit for
# the query "troponin reference interval myocardial infarction" was a garbled index
# page -- "Cardiac troponin (cTnJ assays {Continued}").
_INDEX_PAT = re.compile(
    r",\s*\d{1,4}[a-z]?(?:\s*[-–—]\s*\d{1,4}[a-z]?)?(?=\s*(?:[,;]|$))", re.M)
_INDEX_SEC = re.compile(r"^(?:subject\s+|author\s+|name\s+)?index$", re.I)


def is_index_dump(text: str, min_hits: int = 6) -> bool:
    """Is the chunk a back-of-book index? Density of the 'Term, 12, 15-17, 20t' pattern."""
    hits = len(_INDEX_PAT.findall(text))
    if hits < min_hits:
        return False
    # 2.5+ page references per 400 characters -> an index. The threshold is higher than the
    # reference one (1.5): prose can contain single matches like ", 45", but not that density.
    return hits / max(1.0, len(text) / 400) >= 2.5


def is_low_value(hit: "Hit") -> bool:
    """Not clinical content: a reference dump or a back-of-book index.

    Not dropped but demoted -- still returned if there is no other candidate (see ref-fill).

    For an index, density ALONE is not enough: numeric clinical tables match the same
    pattern too (measured: 97 flags in a 4000-chunk sample, 22 of them not at the end
    of the book, and one was a real carotid stenosis criteria table).
    So density + position are required together; an index lives in the back matter.
    """
    leaf = (hit.section or "").split(">")[-1].strip()
    if leaf and _INDEX_SEC.match(leaf):
        return True
    if is_reference_dump(hit.text):
        return True
    if not is_index_dump(hit.text):
        return False
    if not hit.n_pages:                      # position unknown -> trust the density
        return True
    return hit.page_start / hit.n_pages > 0.85


# ------------------------------------------------------------------------- RRF
def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, list[str]]]:
    acc: dict[int, float] = {}
    src: dict[int, list[str]] = {}
    for name, ids in rankings.items():
        for rank, cid in enumerate(ids):
            acc[cid] = acc.get(cid, 0.0) + 1.0 / (k + rank + 1)
            src.setdefault(cid, []).append(name)
    return sorted(((cid, s, src[cid]) for cid, s in acc.items()),
                  key=lambda t: -t[1])


# -------------------------------------------------------------------- reranker
def rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    global _reranker
    if not hits:
        return hits
    from sentence_transformers import CrossEncoder
    from .embed import get_device, move_fp16
    with _model_lock:
        if _reranker is None:
            dev = get_device()
            gpu = dev in ("mps", "cuda")
            ce = CrossEncoder(RERANKER, device="cpu" if gpu else dev, max_length=512)
            if gpu:
                move_fp16(ce, dev)
            _reranker = ce
        scores = _reranker.predict([(query, h.text) for h in hits],
                                   batch_size=16, show_progress_bar=False)
    for h, s in zip(hits, scores):
        h.score = float(s)
        h.why = h.why + ["ce"]       # mark that the score came from the cross-encoder
    return sorted(hits, key=lambda h: -h.score)[:top_k]


# ----------------------------------------------------------------------- main
def hydrate(con: sqlite3.Connection, scored: list[tuple[int, float, list[str]]]) -> list[Hit]:
    if not scored:
        return []
    order = {cid: i for i, (cid, _, _) in enumerate(scored)}
    q = ",".join("?" * len(order))
    rows = con.execute(
        f"SELECT c.id,b.slug,b.title,c.page_start,c.page_end,c.section,c.text,"
        f"b.n_pages FROM chunks c JOIN books b ON b.id=c.book_id WHERE c.id IN ({q})",
        list(order)).fetchall()
    hits = [Hit(r[0], r[1], r[2], r[3], r[4], r[5] or "", r[6], r[7] or 0)
            for r in rows]
    for h in hits:
        cid, s, why = scored[order[h.chunk_id]]
        h.score, h.why = s, why
    return sorted(hits, key=lambda h: order[h.chunk_id])


def release_device_cache() -> None:
    """Return the MPS cache to the driver once inference is done.

    The MPS allocator keeps activation buffers even after the call ends; with unified
    memory that is RAM directly. Measured (2026-09-30): 986 MB releasable after five
    searches, call time 81 ms. If torch was never loaded (BM25 only), it does not load it.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    with _model_lock:                   # don't race an encode/predict in progress
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def search(con: sqlite3.Connection, query: str, top_k: int = 8,
           book: str | None = None, candidates: int = 30,
           dense: bool = True, use_rerank: bool = True,
           collection: str | None = None) -> list[Hit]:
    try:
        return _search(con, query, top_k, book, candidates, dense, use_rerank, collection)
    finally:
        if dense or use_rerank:
            release_device_cache()


def _search(con: sqlite3.Connection, query: str, top_k: int,
            book: str | None, candidates: int,
            dense: bool, use_rerank: bool,
            collection: str | None) -> list[Hit]:
    rankings = {"bm25": bm25_search(con, query, candidates, book, collection)}
    if dense:
        try:
            d = dense_search(con, query, candidates, book, collection)
            if d:
                rankings["dense"] = d
        except Exception as e:                      # model missing / not downloaded
            rankings.setdefault("_warn", [])
            print(f"[dense atlandi: {e}]", file=sys.stderr, flush=True)

    hits = hydrate(con, rrf({k: v for k, v in rankings.items() if v})[:candidates])
    # Drop exact duplicates (same text in the same book — e.g. repeated pages)
    seen_txt: set = set()
    hits = [h for h in hits
            if (key := (h.book, " ".join(h.text.split())[:200])) not in seen_txt
            and not seen_txt.add(key)]
    # Move reference and index chunks to the end (instead of dropping: sometimes they are the only source)
    low = [is_low_value(h) for h in hits]
    real = [h for h, lo in zip(hits, low) if not lo]
    refs = [h for h, lo in zip(hits, low) if lo]
    hits = real + refs
    if use_rerank and len(hits) > top_k:
        try:
            # rerank runs ONLY on real content; if there are not enough candidates, references
            # are appended AFTER the rerank — the old version fed the reranker a mixed pool
            # and could push references back to the top.
            ranked = rerank(query, real, top_k) if real else []
            fill = refs[:max(0, top_k - len(ranked))]
            for h in fill:
                h.why = h.why + ["ref-fill"]   # RRF-scored filler, not a CE score
            return ranked + fill
        except Exception as e:
            print(f"[rerank atlandi: {e}]", file=sys.stderr, flush=True)
    return hits[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--db", default="corpus.sqlite")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--book")
    ap.add_argument("--collection")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    import time
    t0 = time.time()
    hits = search(con, a.query, a.k, a.book, dense=not a.no_dense,
                  use_rerank=not a.no_rerank, collection=a.collection)
    print(f"({(time.time()-t0)*1000:.0f} ms)\n")
    for h in hits:
        print(f"{h.cite()} {h.score:+.3f} [{'+'.join(h.why)}] {h.section[:60]}")
        print("   " + " ".join(h.text.split())[:220] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
