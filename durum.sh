#!/usr/bin/env bash
# Vektorleme durumu: ilerleme, OLCULEN anlik hiz, kalan sure, surec sagligi
ROOT="$(cd "$(dirname "$0")" && pwd)"
DB="${1:-$ROOT/corpus.sqlite}"
"$ROOT/.venv/bin/python" - "$DB" <<'PY'
import sqlite3, sys, subprocess, time
db = sys.argv[1]
q = lambda: sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute(
    "select (select count(*) from embeddings),(select count(*) from chunks)").fetchone()
v0, n = q()
alive = b"tbmcp.embed" in subprocess.run(["ps", "ax"], capture_output=True).stdout
rate = 0.0
if alive and v0 < n:
    # Is ~512 parcada bir commit eder; sabit 8 sn pencere commit arasina
    # denk gelip 0 gorebiliyordu. Commit yakalayana dek (en cok 45 sn) bekle.
    t0 = time.time()
    while time.time() - t0 < 45:
        time.sleep(3)
        v1 = q()[0]
        if v1 > v0:
            rate = (v1 - v0) / (time.time() - t0)
            break
print(f"vektor {v0:,}/{n:,}  (%{100*v0/n:.1f})   surec: {'calisiyor' if alive else 'DURMUS'}")
if v0 >= n:
    print("BITTI -> 1) scripts/fix_sections.py calistir  2) TEXTBOOK_DENSE=1 yap")
elif alive and rate > 0:
    print(f"olculen hiz {rate:.1f} parca/sn -> ~{(n-v0)/rate/3600:.1f} saat kaldi")
elif alive:
    print("45 sn'de yeni vektor gelmedi — is takilmis olabilir, embed.log'a bak")
else:
    print("surdurmek icin: .venv/bin/python -m tbmcp.embed --db corpus.sqlite")
PY
