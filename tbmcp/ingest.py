#!/usr/bin/env python3
"""PDF ders kitaplarini corpus.sqlite icine al.

Kullanim:
    python -m tbmcp.ingest /kitap/dizini --db corpus.sqlite
    python -m tbmcp.ingest /kitap/dizini --db corpus.sqlite --probe   # sadece rapor, yazma yok

Metin katmani olmayan (taranmis) PDF'ler isaretlenir ve atlanir; ocr.sh onlari isler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf as fitz

SCHEMA = Path(__file__).with_name("schema.sql")

# Metin katmani var mi kararinin esigi: sayfa basina ortalama karakter
SCANNED_THRESHOLD = 200
# Hedef parca boyutu (karakter). ~1200 karakter ~ 300 token.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 180
# Bir satir sayfalarin bu oranindan fazlasinda gecerse ust/alt bilgi sayilir
HEADER_MIN_RATIO = 0.30
HEADER_MAX_LEN = 90


# --------------------------------------------------------------------------- db
def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA.read_text())
    return con


# ---------------------------------------------------------------------- metin
def sha256(path: Path, limit: int = 8 << 20) -> str:
    """Basi + sonu + boyut uzerinden hizli parmak izi (tam dosya degil, kasitli)."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(limit))
        if size > 2 * limit:
            f.seek(-limit, 2)
            h.update(f.read(limit))
    return h.hexdigest()


_HYPHEN = re.compile(r"(\w)[-­]\n(\w)")
_MULTI_NL = re.compile(r"\n{3,}")
_TRAIL_WS = re.compile(r"[ \t]+\n")


def clean(text: str) -> str:
    text = text.replace(" ", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = _HYPHEN.sub(r"\1\2", text)          # satir sonu tirelemesini birlestir
    text = _TRAIL_WS.sub("\n", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def find_running_lines(pages: list[str]) -> set[str]:
    """Sayfalarin cogunda tekrar eden ust/alt bilgi satirlarini bul."""
    if len(pages) < 10:
        return set()
    counter: Counter[str] = Counter()
    for txt in pages:
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        if not lines:
            continue
        for line in lines[:2] + lines[-2:]:
            if len(line) <= HEADER_MAX_LEN:
                # sayfa numaralarini normalize et: "Chapter 4  127" -> "Chapter 4  #"
                counter[re.sub(r"\d+", "#", line)] += 1
    floor = max(3, int(len(pages) * HEADER_MIN_RATIO))
    return {k for k, v in counter.items() if v >= floor}


def strip_running(text: str, running: set[str]) -> str:
    """Tespit edilen ust/alt bilgi satirlarini sayfa kenarlarindan at.

    Kenar penceresi find_running_lines ile AYNI koordinatta olmali: o, BOS OLMAYAN
    satirlarin ilk/son ikisine bakiyor (satir 77). Burasi eskiden ham satir indeksi
    kullaniyordu (i < 2); sayfa numarasi kendi satirinda olup ardindan bos satir
    geldiginde baslik ham indeks 2'ye kayiyor ve elenmiyordu. Olculdu (ders notu koleksiyonu, 6554
    sayfa): basligi kalan sayfa %49.9 -> %0.1, fazladan silinen karakter yalnizca %0.52.
    Ayni hata 88 kitaplik mevcut korpusta da parcalara ustbilgi sizdirmisti.
    """
    if not running:
        return text
    lines = text.split("\n")
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    edge = set(nonempty[:2] + nonempty[-2:])
    keep = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i in edge and s and re.sub(r"\d+", "#", s) in running:
            continue
        keep.append(line)
    return "\n".join(keep)


# --------------------------------------------------------------------- parcala
def section_map(toc: list[tuple[int, str, int]], n_pages: int) -> dict[int, str]:
    """Her sayfa icin 'Bolum > Alt-bolum' yolunu TEK gecisde hesapla.

    Parca basina toc'u bastan taramak O(parca x toc) idi; Mandell'de (4639 toc
    girdisi, 37k parca) bu 170M+ dongu ediyordu. Simdi O(sayfa + toc).
    """
    out: dict[int, str] = {}
    if not toc:
        return out
    # YALNIZ sayfaya gore sirala: Python sort kararli oldugu icin ayni sayfada
    # baslayan girdiler belge (outline) sirasini korur. Eski (page, level)
    # anahtari ayni-sayfa girdilerini yeniden dizip derin basligi yanlis
    # ebeveyne asiyordu (olculdu: 25/43 kitapta ~5.5k sayfa yanlis etiket).
    entries = sorted(toc, key=lambda t: t[2])
    path: dict[int, str] = {}
    i, cur = 0, ""
    for page in range(1, n_pages + 1):
        while i < len(entries) and entries[i][2] <= page:
            level, title, _ = entries[i]
            path[level] = title.strip()
            for deeper in [k for k in path if k > level]:
                del path[deeper]
            cur = " > ".join(path[k] for k in sorted(path))
            i += 1
        if cur:
            out[page] = cur
    return out


def chunk_book(pages: dict[int, str], secmap: dict[int, str]):
    """Kitap metnini sayfa sinirlarini takip ederek parcalara ayir.

    Yield: (page_start, page_end, section, text)
    """
    # Tum metni tek akista birlestir, her karakterin hangi sayfada oldugunu tut
    buf: list[str] = []
    marks: list[tuple[int, int]] = []  # (akistaki_offset, sayfa)
    pos = 0
    for page in sorted(pages):
        txt = pages[page]
        if not txt:
            continue
        marks.append((pos, page))
        buf.append(txt)
        buf.append("\n\n")
        pos += len(txt) + 2
    stream = "".join(buf)
    if not stream.strip():
        return

    def page_at(offset: int) -> int:
        lo, hi, best = 0, len(marks) - 1, marks[0][1]
        while lo <= hi:
            mid = (lo + hi) // 2
            if marks[mid][0] <= offset:
                best = marks[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    start = 0
    n = len(stream)
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        if end < n:
            # paragraf, sonra cumle, sonra bosluk sinirinda kes
            window = stream[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut > CHUNK_CHARS * 0.5:
                    end = start + cut + len(sep)
                    break
        text = stream[start:end].strip()
        if len(text) >= 80:
            p0, p1 = page_at(start), page_at(max(start, end - 1))
            yield p0, p1, secmap.get(p0, ""), text
        if end <= start:
            break
        start = end - CHUNK_OVERLAP if end < n else end


# ----------------------------------------------------------------------- kitap
def slugify(name: str, taken: set[str], strip: tuple[str, ...] = ()) -> str:
    """Dosya adindan kisa, benzersiz bir slug uret.

    strip: slug uretilmeden once addan silinecek duzenli ifadeler (buyuk/kucuk harf
    duyarsiz). Varsayilan bos; dosya adlarinda sabit bir ek ya da kuyruk varsa
    komut satirindan --strip-name ile verilir.
    """
    s = name
    for pattern in strip:
        s = re.sub(pattern, "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    s = re.sub(r"-(pdf|isbn13.*)$", "", s)[:60].strip("-") or "book"
    base, i = s, 2
    while s in taken:
        s, i = f"{base}-{i}", i + 1
    taken.add(s)
    return s


def probe(path: Path) -> dict:
    """Metin katmani var mi? Ilk/orta/son sayfalardan orneklem al."""
    with fitz.open(path) as doc:
        n = doc.page_count
        if n == 0:
            return {"n_pages": 0, "chars_per_page": 0, "scanned": 1, "title": path.stem}
        idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1,
                      min(9, n - 1), min(19, n - 1)})
        total = sum(len(doc[i].get_text("text").strip()) for i in idx)
        cpp = total // len(idx)
        md = doc.metadata or {}
        return {
            "n_pages": n,
            "chars_per_page": cpp,
            "scanned": int(cpp < SCANNED_THRESHOLD),
            "title": (md.get("title") or "").strip() or path.stem,
            "meta": {k: v for k, v in md.items() if v and k in
                     ("author", "producer", "creator", "creationDate")},
        }


def ingest_book(con: sqlite3.Connection, path: Path, slug: str, info: dict,
                collection: str = "textbook", lang: str = "en") -> int:
    with fitz.open(path) as doc:
        raw = [clean(doc[i].get_text("text")) for i in range(doc.page_count)]
        toc = [(lvl, title, max(1, pg)) for lvl, title, pg in (doc.get_toc() or [])
               if pg and pg > 0]

    running = find_running_lines(raw)
    pages = {i + 1: strip_running(t, running) for i, t in enumerate(raw)}

    cur = con.cursor()
    cur.execute(
        "INSERT INTO books(slug,title,path,sha256,n_pages,scanned,chars_per_page,meta,"
        "collection,lang) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (slug, info["title"], str(path), sha256(path), info["n_pages"],
         info["scanned"], info["chars_per_page"], json.dumps(info.get("meta", {})),
         collection, lang),
    )
    book_id = cur.lastrowid
    cur.executemany("INSERT INTO pages(book_id,page,text) VALUES(?,?,?)",
                    ((book_id, p, t) for p, t in pages.items()))
    cur.executemany("INSERT INTO toc(book_id,level,title,page) VALUES(?,?,?,?)",
                    ((book_id, lvl, title, pg) for lvl, title, pg in toc))

    secmap = section_map(toc, info["n_pages"])
    rows = [(book_id, p0, p1, sec, txt) for p0, p1, sec, txt in chunk_book(pages, secmap)]
    cur.executemany(
        "INSERT INTO chunks(book_id,page_start,page_end,section,text) VALUES(?,?,?,?,?)",
        rows)
    # FTS'i ayni transaction icinde artimli doldur: external-content FTS5'te
    # senkron trigger yok; eskiden tek senkron nokta kosu SONUNDAKI rebuild idi —
    # kosu yarida kalirsa commit edilmis kitaplar BM25'te sessizce gorunmuyordu.
    cur.execute(
        "INSERT INTO chunks_fts(rowid,text,section)"
        " SELECT id,text,section FROM chunks WHERE book_id=?", (book_id,))
    con.commit()
    return len(rows)


def rebuild_fts(con: sqlite3.Connection) -> None:
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.commit()


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+", help="PDF dosyasi veya dizini")
    ap.add_argument("--db", default="corpus.sqlite")
    ap.add_argument("--probe", action="store_true",
                    help="yalniz rapor: hangi kitap taranmis, OCR maliyeti ne")
    ap.add_argument("--rebuild-fts", action="store_true",
                    help="FTS indeksini sifirdan kur (onarim; normalde gerekmez)")
    ap.add_argument("--collection", default="textbook", choices=("textbook", "notes"),
                    help="'textbook' (varsayilan) ya da 'notes' (ders notlari; ayri aranir)")
    ap.add_argument("--lang", default="en", help="koleksiyonun dili, orn. en, tr (varsayilan en)")
    ap.add_argument("--strip-name", action="append", default=[], metavar="REGEX",
                    help="slug uretilmeden once dosya adindan silinecek duzenli ifade "
                         "(yinelenebilir), orn. --strip-name='_final$'. Ifade '-' ile "
                         "basliyorsa '=' ile yazmak zorunludur: --strip-name='--\\s*kopya.*$'")
    args = ap.parse_args()
    for pattern in args.strip_name:
        try:
            re.compile(pattern)
        except re.error as e:
            ap.error(f"--strip-name gecersiz duzenli ifade {pattern!r}: {e}")

    pdfs: list[Path] = []
    for s in args.src:
        p = Path(s).expanduser()
        pdfs += sorted(p.rglob("*.pdf")) if p.is_dir() else [p]
    # Ayni dosya iki yoldan gelirse (ic ice dizin, goreli+mutlak) tek kez isle
    _seen: set[str] = set()
    pdfs = [p for p in pdfs if p.is_file()
            and (rp := str(p.resolve())) not in _seen and not _seen.add(rp)]
    if not pdfs:
        print("PDF bulunamadi", file=sys.stderr)
        return 1

    if args.probe:
        print(f"{'kar/sf':>7} {'sayfa':>6}  durum     kitap")
        scanned_pages = 0
        for p in pdfs:
            try:
                info = probe(p)
            except Exception as e:
                print(f"{'ERR':>7} {'-':>6}  hata      {p.name}  ({e})")
                continue
            state = "TARANMIS" if info["scanned"] else "metin-ok"
            if info["scanned"]:
                scanned_pages += info["n_pages"]
            print(f"{info['chars_per_page']:>7} {info['n_pages']:>6}  {state:<9} {p.name[:70]}")
        print(f"\n{len(pdfs)} PDF. OCR gereken sayfa: {scanned_pages}"
              f"  (~{scanned_pages * 1.5 / 3600:.1f} saat @1.5 sn/sayfa)")
        return 0

    con = connect(args.db)
    taken = {r[0] for r in con.execute("SELECT slug FROM books")}
    done = {str(Path(r[0]).resolve())
            for r in con.execute("SELECT path FROM books")}

    t0, n_chunks, n_books, skipped = time.time(), 0, 0, []
    for p in pdfs:
        if str(p.resolve()) in done:
            print(f"  atlandi (zaten var): {p.name[:60]}")
            continue
        try:
            info = probe(p)
        except Exception as e:
            print(f"  HATA {p.name[:60]}: {e}", file=sys.stderr)
            continue
        if info["scanned"]:
            skipped.append(p)
            print(f"  TARANMIS, atlandi: {p.name[:60]} ({info['n_pages']} sayfa)")
            continue
        slug = slugify(p.stem, taken, tuple(args.strip_name))
        try:
            c = ingest_book(con, p, slug, info, args.collection, args.lang)
        except Exception as e:
            # Tek bozuk PDF kosuyu oldurmesin: yarim transaction'i geri al, devam et.
            con.rollback()
            print(f"  HATA {p.name[:60]}: {e} — atlandi", file=sys.stderr)
            continue
        done.add(str(p.resolve()))
        n_chunks += c
        n_books += 1
        print(f"  {slug:<40} {info['n_pages']:>5} sayfa  {c:>6} parca")

    if args.rebuild_fts:
        print("FTS5 indeksi sifirdan kuruluyor...")
        con.rollback()          # olasi yarim transaction'la rebuild'e girme
        rebuild_fts(con)
    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('ingested_at',datetime('now'))")
    con.commit()
    dt = time.time() - t0
    print(f"\n{n_books} kitap, {n_chunks} parca, {dt:.1f} sn")
    if skipped:
        print(f"\nOCR bekleyen {len(skipped)} kitap (bkz. ocr.sh):")
        for p in skipped:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
