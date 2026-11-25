#!/usr/bin/env bash
# Embedding status: progress, MEASURED current rate, time remaining, process health
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
    # The job commits every ~512 chunks; a fixed 8 s window could fall between
    # commits and see 0. Wait until a commit is caught (at most 45 s).
    t0 = time.time()
    while time.time() - t0 < 45:
        time.sleep(3)
        v1 = q()[0]
        if v1 > v0:
            rate = (v1 - v0) / (time.time() - t0)
            break
print(f"vectors {v0:,}/{n:,}  ({100*v0/n:.1f}%)   process: {'running' if alive else 'STOPPED'}")
if v0 >= n:
    print("DONE -> 1) run scripts/fix_sections.py  2) set TEXTBOOK_DENSE=1")
elif alive and rate > 0:
    print(f"measured rate {rate:.1f} chunks/s -> ~{(n-v0)/rate/3600:.1f} hours left")
elif alive:
    print("no new vectors in 45 s — the job may be stuck, check embed.log")
else:
    print("to resume: .venv/bin/python -m tbmcp.embed --db corpus.sqlite")
PY
