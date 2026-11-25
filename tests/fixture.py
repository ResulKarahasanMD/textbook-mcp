"""Small synthetic corpus for tests: the real schema, random normalized fp16 vectors."""
from __future__ import annotations

import os
import sqlite3
import tempfile

import numpy as np

SCHEMA = os.path.join(os.path.dirname(__file__), "..", "tbmcp", "schema.sql")
# slug -> (collection, number of chunks)
KITAPLAR = {"kitap-a": ("textbook", 40), "kitap-b": ("textbook", 60), "not-c": ("notes", 20)}


def vec_blob(v: np.ndarray) -> bytes:
    return (v / np.linalg.norm(v)).astype(np.float16).tobytes()


def korpus(dim: int = 16, seed: int = 7,
           kitaplar: dict[str, tuple[str, int]] | None = None) -> tuple[sqlite3.Connection, str]:
    """(connection, file path). Each chunk's text contains its slug and the word 'insulin'."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    con = sqlite3.connect(path, check_same_thread=False)
    con.executescript(open(SCHEMA, encoding="utf-8").read())
    rng = np.random.default_rng(seed)
    for slug, (koleksiyon, n) in (kitaplar or KITAPLAR).items():
        cur = con.execute("INSERT INTO books(slug,title,path,n_pages,collection,lang) VALUES(?,?,?,?,?,?)",
                          (slug, slug, f"/x/{slug}.pdf", n, koleksiyon, "en"))
        for i in range(n):
            ekle(con, cur.lastrowid, i + 1, f"{slug} parca {i} insulin glucose", rng.standard_normal(dim))
    con.commit()
    return con, path


def ekle(con: sqlite3.Connection, book_id: int, page: int, text: str, v: np.ndarray) -> int:
    cid = con.execute("INSERT INTO chunks(book_id,page_start,page_end,section,text) VALUES(?,?,?,?,?)",
                      (book_id, page, page, "", text)).lastrowid
    con.execute("INSERT INTO chunks_fts(rowid,text,section) VALUES(?,?,?)", (cid, text, ""))
    con.execute("INSERT INTO embeddings(chunk_id,vec) VALUES(?,?)", (cid, vec_blob(v)))
    return cid


def kaba_kuvvet(con: sqlite3.Connection, qv: np.ndarray, limit: int, where: str, params: tuple) -> list[int]:
    """Expected result: dot product over the rows matching the filter, the top `limit` ids by descending score."""
    rows = con.execute("SELECT e.chunk_id, e.vec FROM embeddings e JOIN chunks c ON c.id=e.chunk_id"
                       " JOIN books b ON b.id=c.book_id WHERE " + where, params).fetchall()
    q = qv.astype(np.float32)
    skor = [(float(np.frombuffer(v, dtype=np.float16).astype(np.float32) @ q), cid) for cid, v in rows]
    return [cid for _, cid in sorted(skor, reverse=True)[:limit]]
