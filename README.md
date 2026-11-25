# textbook-mcp

An MCP server that answers clinical questions from **your own** PDF textbooks: hybrid search (SQLite FTS5
BM25 + bge-m3 dense vectors, RRF fusion, cross-encoder reranking) over a single `corpus.sqlite` file, with
page-level citations and a page-image tool for tables. No books, text or vectors are included in this
repository; you index PDFs you are entitled to use. The rest of this README is in Turkish.

An MCP server that answers clinical questions using your own PDF textbooks as its source.
Same division of labor as the DynaMed MCP: **the server searches and retrieves; Claude does the synthesis.**

A single file (`corpus.sqlite`) holds the text, the BM25 index, and the vectors.
No separate vector database, no separate server, no local LLM.

```
PDF  →  ingest  →  corpus.sqlite  →  server (MCP/stdio)  →  Claude
                   ├ pages, toc
                   ├ chunks + FTS5   (BM25, lexical)
                   └ embeddings      (bge-m3, semantic, float16)
```

Search: BM25 ∪ dense → RRF fusion → `bge-reranker-v2-m3` cross-encoder → top-k.

## Installation

```sh
cd textbook-mcp
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dense]"
```

`-e .` installs the package into the venv and generates the `.venv/bin/textbook-mcp` console script —
the MCP registration uses this script (`-m tbmcp.server` depends on the cwd; don't use it).

## 1. Survey the corpus (read-only)

Which books have a text layer and which are scanned (`/kitap/dizini` and `/not/dizini` stand for your
book and notes directories):

```sh
.venv/bin/python -m tbmcp.ingest /kitap/dizini --probe
```

If the average number of characters per page is **< 200**, the book is scanned and needs OCR.

## 2. OCR (scanned books only)

```sh
brew install ocrmypdf             # if not installed (also pulls in tesseract)
brew install tesseract-lang       # if you need languages other than eng
./ocr.sh /kitap/dizini eng
```

The originals are left untouched; output goes to `/kitap/dizini/_ocr/`. About 1.5 s per page.

## 3. Index

```sh
.venv/bin/python -m tbmcp.ingest /kitap/dizini --db corpus.sqlite
.venv/bin/python -m tbmcp.ingest /kitap/dizini/_ocr --db corpus.sqlite   # the OCR'd books
```

A file that is already indexed is skipped; adding books to the corpus later is safe.

There are two collections, searched separately: `textbook` (the default) and `notes` (lecture notes, usually
in a different language from the books). To ingest notes into the second collection:

```sh
.venv/bin/python -m tbmcp.ingest /not/dizini --db corpus.sqlite --collection notes --lang tr
```

The book ID (slug) is derived from the file name. If the file names carry a fixed suffix or tail, remove it
before slugging with `--strip-name=REGEX` (repeatable; if the expression starts with `-`, the `=` form is required):

```sh
.venv/bin/python -m tbmcp.ingest /kitap/dizini --db corpus.sqlite --strip-name='--\s*kopya.*$'
```

The FTS index is updated incrementally with each book; if you suspect it is
corrupted, rebuild it from scratch with `--rebuild-fts`. A single broken PDF does not
kill the run — the error is printed and the remaining books are processed.

## 4. Embed (one-time, MPS)

```sh
.venv/bin/python -m tbmcp.embed --db corpus.sqlite
```

It resumes where it left off — you can interrupt it and continue later.
The server also works without vectors (BM25 only, `TEXTBOOK_DENSE=0`).

## 5. Try it

```sh
.venv/bin/python -m tbmcp.search --db corpus.sqlite "diabetic ketoacidosis insulin infusion"
```

To inspect the layers separately: `--no-dense`, `--no-rerank`.

## 6. Connect to Claude Code

```sh
claude mcp add textbook --scope user \
  -e TEXTBOOK_DB=$PWD/corpus.sqlite \
  -e TEXTBOOK_DENSE=1 -e TEXTBOOK_RERANK=1 \
  -- $PWD/.venv/bin/textbook-mcp
```

If the vectors are not ready yet, start with `TEXTBOOK_DENSE=0`; once `durum.sh` (embedding status) reports
`DONE`, first run `scripts/fix_sections.py`, then re-register with `DENSE=1`.

Recommended workflow: search → open the page → cite with the page number. You can enforce this with a
Claude Code skill (not included in this repository).

## Tools

| Tool | Purpose |
|---|---|
| `textbook_search(query, book?, top_k)` | Hybrid search across the textbooks, with `[book s.N]` citations (`s.` = *sayfa*, page) |
| `textbook_notes_search(query, book?, top_k)` | Search across the lecture notes (the `notes` collection); a separate pool |
| `textbook_get_page(book, page, window)` | Full page + its neighbors |
| `textbook_find_in_book(book, query)` | Within a single book |
| `textbook_toc(book, depth, contains)` | Table of contents (PDF bookmarks) |
| `textbook_list_books()` | Corpus inventory |
| `textbook_page_image(book, page, dpi)` | Page image — **for tables** |
| `textbook_corpus_info()` | Index status |

## Why this design

- **Hybrid, not pure vector.** Medical queries contain terms verbatim ("HbA1c",
  "Ranson"); BM25 finds these better than vectors do. Vectors, in turn, capture paraphrase similarity.
- **The reranker is the single biggest quality gain.** 30 candidates from the first stage → cross-encoder →
  best 8. About 1.5 s per query — measured, and worth paying.
- **bge-m3 is multilingual.** A question asked in Turkish finds English text.
- **No ANN index.** 504k chunks × 1024 dimensions × fp16 = ~1.0 GB; with a numpy matmul a
  query takes ~50 ms. At this scale FAISS/Chroma is unnecessary complexity.
- **Page-image tool.** Dosing tables and staging schemes get mangled when converted
  to plain text; Claude can read the page as an image.

## Measured values (the author's Apple Silicon Mac, MPS)

43 medical textbooks / 90,561 pages / 503,561 chunks, `corpus.sqlite` **1.48 GB**:

| Step | Measurement |
|---|---|
| PDF text extraction | 20 pages/s (disk-bound) → 43 books ~2.5 hours |
| OCR | not needed (43 of the 43 books have a text layer) |
| FTS5 index | ~1 min |
| Embedding (bge-m3, fp16, seq512, batch 32) | 13-18 chunks/s → ~8-10 hours, one-time |
| BM25 query | 1 ms |
| Dense query (500k × 1024 fp16, numpy matmul) | ~50 ms |
| Rerank (30 candidates) | ~1.5 s |
| Model loading (cold start, one-time) | 8 s |

Tuning notes: `max_seq_length=512` + fp16 sped up embedding by 37% (13.3 → 18.3
chunks/s) — since chunks are ~300 tokens, 1024 was already wasteful. On MPS the reranker scales
linearly with the number of candidates (~52 ms/candidate), which is why the candidate count is 30 instead of 50.

### Memory (2026-09-30, 146 books / 1.24 M vectors)

Server process, after five searches followed by a 4 s wait (`/usr/bin/footprint`, phys_footprint;
the measurement script and raw outputs are not in this repository):

| Scenario | Before | After | Peak before → after |
|---|---|---|---|
| Restricted to one book (`textbook_find_in_book`) | 8876 MB | 3398 MB | 8950 → 4613 MB |
| Whole textbook collection (`textbook_search`) | 8872 MB | 5681 MB | 8950 → 6884 MB |
| Lecture notes collection (`textbook_notes_search`) | 8876 MB | 3275 MB | 8952 → 4383 MB |

In 15 of 15 queries the result IDs and scores are identical to the old code. Changes:

- **Models are moved to the device in fp16, tensor by tensor** (`embed.move_fp16`). Loading in fp32 on the
  device and calling `.half()` wasted ~1.1 GB per model on MPS (fp16 tensors in fp32-sized
  buffers: 1682 MiB live, versus the expected 1083).
- **`torch.mps.empty_cache()` after every search** (`search.release_device_cache`, 81 ms): MPS
  was holding ~1 GB of activation buffers. The released memory drops out of the footprint with a
  3–4 s delay.
- **Filtered dense search does not load the whole matrix**: for a single book or a collection of ≤200k rows,
  only those rows are fetched (`search._subset_matrix`; the largest book is 106 MB, the lecture notes
  collection 39 MB). The full matrix (2.5 GB) is loaded only when an unfiltered search arrives.
- **The matrix and subsets live on anonymous mmap** (`embed.empty_f16`): macOS libmalloc holds on to freed
  large blocks (a deleted 2.3 GB stayed in the footprint); freeing via mmap actually
  returns it. Scoring also uses a fixed 8 MB buffer instead of 268 MB temporaries.

Known residue: after idle unloading the process stays at 390–980 MB (old code 373–944 MB).
The difference comes from small/medium blocks (MALLOC_SMALL) allocated and released while the models load
staying in the allocator; `malloc_zone_pressure_relief` does not reclaim them. Getting it to zero would require
running the models in a subprocess.

Tests: `.venv/bin/python -m unittest discover -s tests -t .` (no models required);
for the GPU memory tests, prefix with `TEXTBOOK_GPU_TESTS=1` (~20 s, loads both models).

## Limitations

- Text on OCR'd pages may contain errors; for critical numbers, verify against the original with
  `textbook_page_image`.
- **Turkish queries do not work until the vectors are ready.** BM25 is lexical: the query "preeklampside
  magnezyum sulfat dozu" (magnesium sulfate dose in preeclampsia) returns zero results against English text.
  Cross-lingual search becomes available once the dense layer is complete. Until then, write your queries in English.
- `page` is everywhere the **PDF page number**, not the book's printed page number.
  (The `books.page_offset` field is reserved for this but not used yet.)
- The table of contents comes from PDF bookmarks; for a book without bookmarks `textbook_toc` returns nothing.
- A textbook ≠ a current guideline. For treatment recommendations, check the edition year and
  cross-check with DynaMed.
- **The vector and rerank paths have not been re-tested in this release.** What was run after the changes made
  for the release: the unit tests that need no models, and end-to-end indexing and search with BM25 only. The
  three GPU memory tests that load both models (`bge-m3`, `bge-reranker-v2-m3`) and dense/rerank search were not
  run; the measurements above were taken with the private version.
- This repository contains no corpus: the books, `corpus.sqlite`, chunks and vectors are not published. Index only
  PDFs you have the right to use; the user is responsible for the content and its copyright.

## License

Code: [MIT](LICENSE).

