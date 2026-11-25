#!/usr/bin/env bash
# Add a text layer to scanned PDFs (ocrmypdf + tesseract).
#
#   ./ocr.sh <pdf-dir> [lang]           # lang: eng (default), tur, "eng+tur"
#
# The original files are NOT TOUCHED. Output: <dizin>/_ocr/<ad>.pdf (<directory>/_ocr/<name>.pdf)
# Then: python -m tbmcp.ingest <dizin>/_ocr --db corpus.sqlite
set -euo pipefail

SRC="${1:?usage: ./ocr.sh <pdf-dir> [lang]}"
LANG="${2:-eng}"
OUT="$SRC/_ocr"
JOBS="${JOBS:-3}"          # number of books in parallel; each already uses many cores

command -v ocrmypdf >/dev/null || { echo "ocrmypdf not found: brew install ocrmypdf"; exit 1; }
tesseract --list-langs 2>/dev/null | grep -qx "${LANG%%+*}" || {
  echo "no tesseract language data: ${LANG%%+*}"
  echo "install: brew install tesseract-lang"; exit 1; }

mkdir -p "$OUT"

ocr_one() {
  local f="$1" out="$2" lang="$3"
  local dst="$out/$(basename "$f")"
  [ -s "$dst" ] && { echo "  skipped (exists): $(basename "$f")"; return 0; }
  echo "  OCR: $(basename "$f")"
  # --skip-text: leave pages that have a text layer as they are (for mixed books)
  # --optimize 1: keep the file size under control, disk 37 GB
  ocrmypdf --skip-text --language "$lang" --optimize 1 --quiet \
           --rotate-pages --deskew "$f" "$dst.part" \
    && mv "$dst.part" "$dst" \
    || { echo "  FAILED: $(basename "$f")" >&2; rm -f "$dst.part"; }
}
export -f ocr_one

echo "source: $SRC   output: $OUT   lang: $LANG   parallel: $JOBS"
/usr/bin/find "$SRC" -maxdepth 1 -iname '*.pdf' -print0 \
  | xargs -0 -P "$JOBS" -I{} bash -c 'ocr_one "$@"' _ {} "$OUT" "$LANG"

echo
echo "done. now:  python -m tbmcp.ingest '$OUT' --db corpus.sqlite"
