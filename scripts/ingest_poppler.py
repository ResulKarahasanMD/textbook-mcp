#!/usr/bin/env python3
"""MuPDF'in patolojik yavas kaldigi PDF'i poppler(pdftotext) ciktisiyla korpusa al.

Nicin: bazi bozuk/onarilmis PDF'lerde MuPDF sayfa icerigi parse'i ~0.4 sayfa/sn'ye
duser (olculdu: 2 ciltlik bir ders kitabinda, qpdf onarimi sonrasi bile). Poppler ayni dosyayi
~200 sayfa/sn okur. Bu betik pdftotext ciktisini (sayfa siniri = form-feed \\f) ve
istege bagli fitz TOC json'unu alir, tbmcp.ingest'in temizlik/parcalama/FTS yolunu
aynen kullanir.

    pdftotext kitap.pdf kitap.txt
    .venv/bin/python scripts/ingest_poppler.py --pdf kitap.pdf --txt kitap.txt \\
        [--toc toc.json] [--title "..."] [--db corpus.sqlite]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tbmcp.ingest import (connect, clean, find_running_lines, strip_running,
                          chunk_book, section_map, sha256, slugify)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--txt", required=True)
    ap.add_argument("--toc", default=None, help="fitz get_toc() json dokumu")
    ap.add_argument("--title", default=None)
    ap.add_argument("--slug", default=None,
                    help="acik slug; verilmezse pdf adindan turetilir "
                         "(dosya adindaki ekler slug'a sizmasin diye acik verilebilir)")
    ap.add_argument("--collection", default="textbook", help="'textbook' | 'notes'")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--db", default=str(ROOT / "corpus.sqlite"))
    a = ap.parse_args()

    pdf = Path(a.pdf)
    raw = Path(a.txt).read_text(encoding="utf-8", errors="replace").split("\f")
    if raw and not raw[-1].strip():
        raw.pop()                          # pdftotext son sayfadan sonra da \f basar
    n_pages = len(raw)
    toc = []
    if a.toc:
        toc = [(lvl, t, max(1, pg)) for lvl, t, pg in json.load(open(a.toc))
               if pg and pg > 0]

    raw = [clean(t) for t in raw]
    running = find_running_lines(raw)
    pages = {i + 1: strip_running(t, running) for i, t in enumerate(raw)}
    cpp = sum(len(t) for t in pages.values()) // max(1, n_pages)

    con = connect(a.db)
    if con.execute("SELECT 1 FROM books WHERE path=?", (str(pdf),)).fetchone():
        print(f"zaten var: {pdf}"); return 0
    taken = {r[0] for r in con.execute("SELECT slug FROM books")}
    slug = a.slug or slugify(pdf.stem, taken)
    if slug in taken:
        print(f"slug zaten kullanimda: {slug}"); return 1
    title = a.title or pdf.stem
    meta = {"extractor": "poppler-pdftotext"}
    if a.collection == "textbook":
        meta["note"] = ("mupdf sayfa parse'i patolojik yavas; "
                        "bkz. scripts/ingest_poppler.py")

    cur = con.cursor()
    cur.execute(
        "INSERT INTO books(slug,title,path,sha256,n_pages,scanned,chars_per_page,meta,"
        "collection,lang) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (slug, title, str(pdf), sha256(pdf), n_pages, 0, cpp, json.dumps(meta),
         a.collection, a.lang))
    book_id = cur.lastrowid
    cur.executemany("INSERT INTO pages(book_id,page,text) VALUES(?,?,?)",
                    ((book_id, p, t) for p, t in pages.items()))
    cur.executemany("INSERT INTO toc(book_id,level,title,page) VALUES(?,?,?,?)",
                    ((book_id, lvl, t, pg) for lvl, t, pg in toc))
    secmap = section_map(toc, n_pages)
    rows = [(book_id, p0, p1, sec, txt) for p0, p1, sec, txt in chunk_book(pages, secmap)]
    cur.executemany(
        "INSERT INTO chunks(book_id,page_start,page_end,section,text) VALUES(?,?,?,?,?)",
        rows)
    cur.execute(
        "INSERT INTO chunks_fts(rowid,text,section)"
        " SELECT id,text,section FROM chunks WHERE book_id=?", (book_id,))
    cur.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('ingested_at',datetime('now'))")
    con.commit()
    print(f"{slug}: {n_pages} sayfa, {len(rows)} parca, {cpp} kar/sayfa, "
          f"toc {len(toc)}, [{a.collection}/{a.lang}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
