# textbook-mcp

An MCP server that answers clinical questions from **your own** PDF textbooks: hybrid search (SQLite FTS5
BM25 + bge-m3 dense vectors, RRF fusion, cross-encoder reranking) over a single `corpus.sqlite` file, with
page-level citations and a page-image tool for tables. No books, text or vectors are included in this
repository; you index PDFs you are entitled to use. The rest of this README is in Turkish.

Kendi PDF ders kitaplarını kaynak alarak klinik soru yanıtlayan MCP sunucusu.
DynaMed MCP ile aynı iş bölümü: **sunucu arar ve getirir, sentezi Claude yapar.**

Tek dosya (`corpus.sqlite`) içinde hem metin, hem BM25 indeksi, hem vektörler.
Ayrı vektör veritabanı, ayrı sunucu, yerel LLM yok.

```
PDF  →  ingest  →  corpus.sqlite  →  server (MCP/stdio)  →  Claude
                   ├ pages, toc
                   ├ chunks + FTS5   (BM25, leksik)
                   └ embeddings      (bge-m3, anlamsal, float16)
```

Arama: BM25 ∪ dense → RRF birleştirme → `bge-reranker-v2-m3` cross-encoder → top-k.

## Kurulum

```sh
cd textbook-mcp
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dense]"
```

`-e .` paketi venv'e kurar ve `.venv/bin/textbook-mcp` konsol betiğini üretir —
MCP kaydı bu betiği kullanır (`-m tbmcp.server` cwd'ye bağımlıdır, kullanma).

## 1. Korpusu tara (yazmadan)

Hangi kitapların metin katmanı var, hangileri taranmış:

```sh
.venv/bin/python -m tbmcp.ingest /kitap/dizini --probe
```

Sayfa başına ortalama karakter **< 200** ise o kitap taranmıştır, OCR gerekir.

## 2. OCR (yalnız taranmış kitaplar için)

```sh
brew install ocrmypdf             # kurulu değilse (tesseract'ı da getirir)
brew install tesseract-lang       # eng dışında dil gerekiyorsa
./ocr.sh /kitap/dizini eng
```

Orijinallere dokunmaz; çıktı `/kitap/dizini/_ocr/`. Sayfa başına ~1.5 sn.

## 3. İndeksle

```sh
.venv/bin/python -m tbmcp.ingest /kitap/dizini --db corpus.sqlite
.venv/bin/python -m tbmcp.ingest /kitap/dizini/_ocr --db corpus.sqlite   # OCR'lananlar
```

Aynı dosya ikinci kez atlanır; korpusa sonradan kitap eklemek güvenlidir.

İki koleksiyon vardır ve ayrı aranır: `textbook` (varsayılan) ve `notes` (ders notları, genellikle
kitaplardan farklı bir dilde). Notları ikinci koleksiyona almak için:

```sh
.venv/bin/python -m tbmcp.ingest /not/dizini --db corpus.sqlite --collection notes --lang tr
```

Kitap kimliği (slug) dosya adından üretilir. Dosya adlarında sabit bir ek ya da kuyruk varsa
`--strip-name=REGEX` ile slug'dan önce silinir (yinelenebilir; ifade `-` ile başlıyorsa `=` biçimi zorunludur):

```sh
.venv/bin/python -m tbmcp.ingest /kitap/dizini --db corpus.sqlite --strip-name='--\s*kopya.*$'
```

FTS indeksi her kitapla birlikte artımlı güncellenir; bozulduğundan
şüphelenirsen `--rebuild-fts` ile sıfırdan kur. Tek bozuk PDF koşuyu
öldürmez — hata basılır, kalan kitaplar işlenir.

## 4. Vektörle (tek seferlik, MPS)

```sh
.venv/bin/python -m tbmcp.embed --db corpus.sqlite
```

Kaldığı yerden devam eder — yarıda kesip sonra sürdürebilirsin.
Vektörler olmadan da sunucu çalışır (yalnız BM25, `TEXTBOOK_DENSE=0`).

## 5. Dene

```sh
.venv/bin/python -m tbmcp.search --db corpus.sqlite "diabetic ketoacidosis insulin infusion"
```

Katmanları ayrı ayrı görmek için: `--no-dense`, `--no-rerank`.

## 6. Claude Code'a bağla

```sh
claude mcp add textbook --scope user \
  -e TEXTBOOK_DB=$PWD/corpus.sqlite \
  -e TEXTBOOK_DENSE=1 -e TEXTBOOK_RERANK=1 \
  -- $PWD/.venv/bin/textbook-mcp
```

Vektörler henüz hazır değilse `TEXTBOOK_DENSE=0` ile başlat; `durum.sh` BİTTİ
deyince önce `scripts/fix_sections.py` çalıştır, sonra kaydı `DENSE=1` ile yenile.

Önerilen iş akışı: ara → sayfayı aç → sayfa numarasıyla alıntıla. Bunu bir Claude Code skill'i ile
zorlayabilirsin (bu depoda yok).

## Araçlar

| Araç | İş |
|---|---|
| `textbook_search(query, book?, top_k)` | Ders kitaplarında hibrit arama, `[kitap s.N]` künyeli |
| `textbook_notes_search(query, book?, top_k)` | Ders notlarında (`notes` koleksiyonu) arama; ayrı havuz |
| `textbook_get_page(book, page, window)` | Tam sayfa + komşuları |
| `textbook_find_in_book(book, query)` | Tek kitap içinde |
| `textbook_toc(book, depth, contains)` | İçindekiler (PDF yer imleri) |
| `textbook_list_books()` | Korpus envanteri |
| `textbook_page_image(book, page, dpi)` | Sayfa görüntüsü — **tablolar için** |
| `textbook_corpus_info()` | İndeks durumu |

## Neden bu tasarım

- **Hibrit, saf vektör değil.** Tıp sorgularında terimler birebir geçer ("HbA1c",
  "Ranson"); BM25 bunları vektörden iyi bulur. Vektör de ifade benzerliğini yakalar.
- **Reranker en büyük tek kalite kazancı.** İlk aşamadan 30 aday → cross-encoder →
  en iyi 8. Sorgu başına ~1,5 sn — ölçüldü, ödemeye değer.
- **bge-m3 çok dilli.** Türkçe sorulan soru İngilizce metni bulur.
- **ANN indeksi yok.** 504k parça × 1024 boyut × fp16 = ~1,0 GB; numpy matmul ile
  sorgu ~50 ms. Bu ölçekte FAISS/Chroma gereksiz karmaşıklık.
- **Sayfa görüntüsü aracı.** Doz tabloları ve evreleme şemaları düz metne
  çevrildiğinde bozulur; Claude sayfayı görüntü olarak okuyabilir.

## Ölçülen değerler (yazarın Apple Silicon Mac'i, MPS)

43 tıp ders kitabı / 90.561 sayfa / 503.561 parça, `corpus.sqlite` **1,48 GB**:

| Adım | Ölçüm |
|---|---|
| PDF metin çıkarma | 20 sayfa/sn (disk-bağımlı) → 43 kitap ~2,5 saat |
| OCR | gerekmedi (43 kitabın 43'ünde metin katmanı var) |
| FTS5 indeksi | ~1 dk |
| Embedding (bge-m3, fp16, seq512, batch 32) | 13-18 parça/sn → ~8-10 saat, tek sefer |
| BM25 sorgusu | 1 ms |
| Dense sorgu (500k × 1024 fp16, numpy matmul) | ~50 ms |
| Rerank (30 aday) | ~1,5 sn |
| Model yükleme (soğuk başlangıç, tek sefer) | 8 sn |

Ayar notları: `max_seq_length=512` + fp16 embedding'i %37 hızlandırdı (13,3 → 18,3
parça/sn) — parçalar ~300 token olduğu için 1024 zaten israftı. Reranker MPS'te
aday sayısıyla doğru orantılı (~52 ms/aday), bu yüzden aday sayısı 50 yerine 30.

### Bellek (2026-09-30, 146 kitap / 1,24 M vektör)

Sunucu süreci, beş arama sonrası 4 sn beklenerek (`/usr/bin/footprint`, phys_footprint;
ölçüm betiği ve ham çıktılar bu depoda yok):

| Senaryo | Önce | Sonra | Zirve önce → sonra |
|---|---|---|---|
| Tek kitapla sınırlı (`textbook_find_in_book`) | 8876 MB | 3398 MB | 8950 → 4613 MB |
| Tüm ders kitabı koleksiyonu (`textbook_search`) | 8872 MB | 5681 MB | 8950 → 6884 MB |
| Ders notu koleksiyonu (`textbook_notes_search`) | 8876 MB | 3275 MB | 8952 → 4383 MB |

15 sorgunun 15'inde sonuç kimlikleri ve skorları eski kodla aynı. Değişiklikler:

- **Modeller cihaza tensör tensör fp16 taşınıyor** (`embed.move_fp16`). Cihazda fp32 yükleyip
  `.half()` çağırmak MPS'te model başına ~1,1 GB boşa harcıyordu (fp16 tensörler fp32 boylu
  tamponlarda: canlı 1682 MiB, olması gereken 1083).
- **Her aramadan sonra `torch.mps.empty_cache()`** (`search.release_device_cache`, 81 ms): MPS
  aktivasyon tamponlarını ~1 GB tutuyordu. Geri verilen bellek ayak izinden 3–4 sn
  gecikmeyle düşüyor.
- **Filtreli dense arama tüm matrisi yüklemiyor**: tek kitap ya da ≤200k satırlık koleksiyon
  için yalnız o satırlar çekiliyor (`search._subset_matrix`; en büyük kitap 106 MB, ders notu
  koleksiyonu 39 MB). Tüm matris (2,5 GB) ancak filtresiz arama gelince yükleniyor.
- **Matris ve alt kümeler anonim mmap üzerinde** (`embed.empty_f16`): macOS libmalloc bırakılan
  büyük blokları tutuyor (silinmiş 2,3 GB ayak izinde kalıyordu); mmap'le boşaltma gerçekten
  geri veriyor. Puanlama da 268 MB'lık geçiciler yerine 8 MB'lık sabit tamponla yapılıyor.

Bilinen kalıntı: boşta boşaltmadan sonra süreç 390–980 MB'ta kalıyor (eski kod 373–944 MB).
Fark, model yüklenirken ayrılıp bırakılan küçük/orta blokların (MALLOC_SMALL) ayırıcıda
kalmasından; `malloc_zone_pressure_relief` geri almıyor. Sıfırlamak için modellerin alt
süreçte çalışması gerekir.

Testler: `.venv/bin/python -m unittest discover -s tests -t .` (model gerektirmez);
GPU bellek testleri için başına `TEXTBOOK_GPU_TESTS=1` (~20 sn, iki modeli yükler).

## Sınırlar

- OCR'lı sayfalarda metin hatalı olabilir; kritik sayılarda `textbook_page_image`
  ile aslını doğrula.
- **Vektörler hazır olana kadar Türkçe sorgu çalışmaz.** BM25 leksiktir: "preeklampside
  magnezyum sulfat dozu" sorgusu İngilizce metinde sıfır sonuç verir. Dense katman
  tamamlanınca çapraz dilli arama açılır. O zamana dek sorguları İngilizce yaz.
- `page` her yerde **PDF sayfa numarasıdır**, kitabın basılı sayfa numarası değil.
  (`books.page_offset` alanı bunun için ayrıldı, henüz kullanılmıyor.)
- İçindekiler PDF yer imlerinden gelir; yer imi olmayan kitapta `textbook_toc` boş döner.
- Ders kitabı ≠ güncel kılavuz. Tedavi önerilerinde baskı yılına bak, DynaMed ile
  çapraz kontrol et.
- **Vektör ve rerank yolu bu yayın sürümünde yeniden sınanmadı.** Yayın için yapılan değişikliklerden sonra
  koşulanlar: model gerektirmeyen birim testleri ve yalnız BM25 ile uçtan uca indeksleme ve arama. İki modeli
  (`bge-m3`, `bge-reranker-v2-m3`) yükleyen üç GPU bellek testi ve dense/rerank araması koşulmadı; yukarıdaki
  ölçümler özel sürümle alınmıştır.
- Bu depoda korpus yok: kitaplar, `corpus.sqlite`, parçalar ve vektörler yayımlanmaz. Yalnız kullanma hakkın olan
  PDF'leri indeksle; içerikten ve telif haklarından kullanıcı sorumludur.

## Lisans

Kod: [MIT](LICENSE).

