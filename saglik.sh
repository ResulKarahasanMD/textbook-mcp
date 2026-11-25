#!/usr/bin/env bash
# Korpus saglik raporu — markdown olarak stdout'a. Salt okur, hicbir sey yazmaz.
#
#   ./saglik.sh                      # ekrana
#   ./saglik.sh > saglik-raporu-$(date +%F).md
#
# Elle yazilan saglik-raporu-*.md dosyalarinin yerine gecer: onlar bayatliyordu
# (2026-09-21 tarihlisi 88 kitap / %100 vektor diyordu, gercek 146 kitapti).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DB="${1:-$ROOT/corpus.sqlite}"

"$ROOT/.venv/bin/python" - "$DB" <<'PY'
import os, sqlite3, subprocess, sys, shutil
from datetime import datetime

db = sys.argv[1]
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
one = lambda s, a=(): con.execute(s, a).fetchone()[0]

print(f"# Korpus saglik raporu — {datetime.now():%Y-%m-%d %H:%M}\n")
print(f"`{db}` — {os.path.getsize(db)/2**30:.1f} GiB\n")

# ---------------------------------------------------------------- olcekler
print("## Olcek\n")
print("| koleksiyon | kitap | sayfa | parca | vektor | kapsam |")
print("|---|---:|---:|---:|---:|---:|")
for (coll,) in con.execute("SELECT DISTINCT collection FROM books ORDER BY 1"):
    k, s = one("SELECT count(*) FROM books WHERE collection=?", (coll,)), \
           one("SELECT coalesce(sum(n_pages),0) FROM books WHERE collection=?", (coll,))
    p = one("SELECT count(*) FROM chunks c JOIN books b ON b.id=c.book_id"
            " WHERE b.collection=?", (coll,))
    v = one("SELECT count(*) FROM chunks c JOIN books b ON b.id=c.book_id"
            " JOIN embeddings e ON e.chunk_id=c.id WHERE b.collection=?", (coll,))
    print(f"| `{coll}` | {k} | {s:,} | {p:,} | {v:,} | %{100*v/max(1,p):.1f} |")
tp, tv = one("SELECT count(*) FROM chunks"), one("SELECT count(*) FROM embeddings")
print(f"| **toplam** | **{one('SELECT count(*) FROM books')}** |"
      f" **{one('SELECT coalesce(sum(n_pages),0) FROM books'):,}** |"
      f" **{tp:,}** | **{tv:,}** | **%{100*tv/max(1,tp):.1f}** |\n")

# -------------------------------------------------------------- butunluk
print("## Butunluk\n")
checks = [
    ("quick_check", one("PRAGMA quick_check"), "ok"),
    ("parcasiz kitap", one("SELECT count(*) FROM books b WHERE NOT EXISTS"
                           " (SELECT 1 FROM chunks c WHERE c.book_id=b.id)"), 0),
    ("vektorsuz parca", tp - tv, 0),
    ("FTS satir farki", one("SELECT (SELECT count(*) FROM chunks_fts)"
                            "-(SELECT count(*) FROM chunks)"), 0),
    ("taranmis (OCR bekleyen)", one("SELECT count(*) FROM books WHERE scanned=1"), 0),
    ("oksuz vektor", one("SELECT count(*) FROM embeddings e WHERE NOT EXISTS"
                         " (SELECT 1 FROM chunks c WHERE c.id=e.chunk_id)"), 0),
]
print("| kontrol | deger | beklenen | |")
print("|---|---:|---:|:--:|")
for ad, deger, bek in checks:
    print(f"| {ad} | {deger} | {bek} | {'OK' if deger == bek else 'DIKKAT'} |")

# ------------------------------------------------------- page_image / TOC
pdf_var = pdf_yok = goreli = 0
for (p,) in con.execute("SELECT path FROM books"):
    if not os.path.isabs(p):
        goreli += 1
    elif os.path.exists(p):
        pdf_var += 1
    else:
        pdf_yok += 1
toc_yok = one("SELECT count(*) FROM books b WHERE NOT EXISTS"
              " (SELECT 1 FROM toc t WHERE t.book_id=b.id)")
toc_zayif = one("SELECT count(*) FROM (SELECT b.id FROM books b LEFT JOIN toc t"
                " ON t.book_id=b.id GROUP BY b.id HAVING count(t.book_id) BETWEEN 1 AND 4)")
print(f"\n## Kaynak PDF ve TOC\n")
print(f"- `page_image` **calisir**: {pdf_var} kitap (PDF yerinde)")
print(f"- calismaz — dosya yok: {pdf_yok}")
print(f"- calismaz — `books.path` **goreli**, sunucu cwd'sine gore cozuluyor: {goreli}")
print(f"- TOC hic yok: {toc_yok} kitap  |  TOC zayif (1-4 kayit): {toc_zayif}")
if toc_yok:
    kt = [r[0] for r in con.execute(
        "SELECT b.slug FROM books b WHERE NOT EXISTS"
        " (SELECT 1 FROM toc t WHERE t.book_id=b.id) ORDER BY 1")]
    print(f"  - {', '.join(kt)}")

# ------------------------------------------------------------------ sistem
print("\n## Sistem\n")
du = shutil.disk_usage(os.path.dirname(db) or ".")
swap = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                      capture_output=True, text=True).stdout.strip()
wal = db + "-wal"
print(f"- disk: {du.free/2**30:.0f} GiB bos / {du.total/2**30:.0f} GiB")
print(f"- takas: {swap}")
if os.path.exists(wal):
    print(f"- WAL: {os.path.getsize(wal)/2**20:.0f} MiB"
          f"{'  (buyuk — `PRAGMA wal_checkpoint(TRUNCATE)` dusun)' if os.path.getsize(wal) > 100*2**20 else ''}")
alive = b"tbmcp.embed" in subprocess.run(["ps", "ax"], capture_output=True).stdout
print(f"- embed sureci: {'CALISIYOR' if alive else 'yok'}")
em = con.execute("SELECT v FROM meta WHERE k='embed_model'").fetchone()
print(f"- embed modeli: {em[0] if em else '(kayitli degil)'}")
con.close()
PY
