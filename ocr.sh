#!/usr/bin/env bash
# Taranmis PDF'lere metin katmani ekle (ocrmypdf + tesseract).
#
#   ./ocr.sh /kitap/dizini [dil]        # dil: eng (varsayilan), tur, "eng+tur"
#
# Orijinal dosyalara DOKUNULMAZ. Cikti: <dizin>/_ocr/<ad>.pdf
# Sonra: python -m tbmcp.ingest <dizin>/_ocr --db corpus.sqlite
set -euo pipefail

SRC="${1:?kullanim: ./ocr.sh /kitap/dizini [dil]}"
LANG="${2:-eng}"
OUT="$SRC/_ocr"
JOBS="${JOBS:-3}"          # es zamanli kitap sayisi; her biri zaten cok cekirdek kullanir

command -v ocrmypdf >/dev/null || { echo "ocrmypdf yok: brew install ocrmypdf"; exit 1; }
tesseract --list-langs 2>/dev/null | grep -qx "${LANG%%+*}" || {
  echo "tesseract dil verisi yok: ${LANG%%+*}"
  echo "kur: brew install tesseract-lang"; exit 1; }

mkdir -p "$OUT"

ocr_one() {
  local f="$1" out="$2" lang="$3"
  local dst="$out/$(basename "$f")"
  [ -s "$dst" ] && { echo "  atlandi (var): $(basename "$f")"; return 0; }
  echo "  OCR: $(basename "$f")"
  # --skip-text: metin katmani olan sayfalari oldugu gibi birak (karma kitaplar icin)
  # --optimize 1: dosya boyutunu kontrol altinda tut, disk 37 GB
  ocrmypdf --skip-text --language "$lang" --optimize 1 --quiet \
           --rotate-pages --deskew "$f" "$dst.part" \
    && mv "$dst.part" "$dst" \
    || { echo "  BASARISIZ: $(basename "$f")" >&2; rm -f "$dst.part"; }
}
export -f ocr_one

echo "kaynak: $SRC   cikti: $OUT   dil: $LANG   paralel: $JOBS"
/usr/bin/find "$SRC" -maxdepth 1 -iname '*.pdf' -print0 \
  | xargs -0 -P "$JOBS" -I{} bash -c 'ocr_one "$@"' _ {} "$OUT" "$LANG"

echo
echo "bitti. simdi:  python -m tbmcp.ingest '$OUT' --db corpus.sqlite"
