#!/usr/bin/env python3
"""Recompute the section labels with the corrected section_map.

Why: the old section_map reordered TOC entries starting on the same page by the
(page, level) key; a deep heading got hung under the wrong parent.
Audit (2026-08-26): ~18.6k chunks wrong, ~11.5k chunks carry a stale label.

WHEN: AFTER the embed job has FINISHED (the FTS rebuild holds the write lock for minutes,
and a running embed dies with 'database is locked'). The script stops if it sees a running embed.

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
    ap.add_argument("--force", action="store_true", help="skip the running-embed check")
    a = ap.parse_args()

    if not a.force and b"tbmcp.embed" in subprocess.run(["ps", "ax"], capture_output=True).stdout:
        print("ERROR: tbmcp.embed is running. Wait for it to finish (status.sh) or use --force.",
              file=sys.stderr)
        return 1

    con = sqlite3.connect(a.db)
    con.execute("PRAGMA busy_timeout=30000")
    t0, updated, books = time.time(), 0, 0
    for bid, slug, n_pages in con.execute("SELECT id,slug,n_pages FROM books").fetchall():
        # ORDER BY id = outline (document) pre-order; section_map keeps it with a stable sort
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
            print(f"  {slug[:50]:<52} {len(diff):>6} chunks to fix")
            if not a.dry_run:
                con.executemany("UPDATE chunks SET section=? WHERE id=?", diff)
                con.commit()
    print(f"\n{updated} chunks in {books} books {'would be fixed (dry-run)' if a.dry_run else 'fixed'}")
    if not a.dry_run and updated:
        print("Rebuilding the FTS index (section is indexed — required)...")
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        con.commit()
        print(f"done: {time.time()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
