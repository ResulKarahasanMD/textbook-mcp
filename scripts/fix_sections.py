#!/usr/bin/env python3
"""Bolum (section) etiketlerini duzeltilmis section_map ile yeniden hesapla.

Neden: eski section_map ayni sayfada baslayan TOC girdilerini (page, level)
anahtariyla yeniden siraliyordu; derin baslik yanlis ebeveynin altina asiliyordu.
Denetim (2026-08-26): ~18.6k parca hatali, ~11.5k parca bayat etiket tasiyor.

NE ZAMAN: embed isi BITTIKTEN sonra (FTS rebuild yazma kilidini dakikalarca tutar,
kosan embed 'database is locked' ile olur). Betik kosan embed gorurse durur.

    .venv/bin/python scripts/fix_sections.py [--db corpus.sqlite] [--dry-run]
"""
from __future__ import annotations
import argparse, subprocess, sqlite3, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from tbmcp.ingest import section_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "corpus.sqlite"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="kosan embed kontrolunu atla")
    a = ap.parse_args()

    if not a.force and b"tbmcp.embed" in subprocess.run(["ps", "ax"], capture_output=True).stdout:
        print("HATA: tbmcp.embed kosuyor. Once bitmesini bekle (durum.sh) veya --force.",
              file=sys.stderr)
        return 1

    con = sqlite3.connect(a.db)
    con.execute("PRAGMA busy_timeout=30000")
    t0, updated, books = time.time(), 0, 0
    for bid, slug, n_pages in con.execute("SELECT id,slug,n_pages FROM books").fetchall():
        # ORDER BY id = outline (belge) on-sirasi; section_map kararli sort'la bunu korur
        toc = [(lvl, (t or "").replace("\r", " ").strip(), pg) for lvl, t, pg in con.execute(
            "SELECT level,title,page FROM toc WHERE book_id=? ORDER BY id", (bid,))]
        secmap = section_map(toc, n_pages)
        rows = con.execute(
            "SELECT id,page_start,section FROM chunks WHERE book_id=?", (bid,)).fetchall()
        diff = [(secmap.get(ps, ""), cid) for cid, ps, old in rows
                if secmap.get(ps, "") != (old or "")]
        if diff:
            books += 1
            updated += len(diff)
            print(f"  {slug[:50]:<52} {len(diff):>6} parca duzeltilecek")
            if not a.dry_run:
                con.executemany("UPDATE chunks SET section=? WHERE id=?", diff)
                con.commit()
    print(f"\n{books} kitapta {updated} parca {'duzeltilirdi (dry-run)' if a.dry_run else 'duzeltildi'}")
    if not a.dry_run and updated:
        print("FTS indeksi yeniden kuruluyor (section indeksli — zorunlu)...")
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        con.commit()
        print(f"bitti: {time.time()-t0:.0f} sn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
