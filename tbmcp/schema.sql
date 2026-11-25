PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS books(
  id          INTEGER PRIMARY KEY,
  slug        TEXT UNIQUE NOT NULL,
  title       TEXT NOT NULL,
  path        TEXT UNIQUE NOT NULL,
  sha256      TEXT,
  n_pages     INTEGER,
  page_offset INTEGER DEFAULT 0,   -- basili sayfa no = pdf sayfa no - page_offset
  scanned     INTEGER DEFAULT 0,   -- 1 ise metin katmani yok, OCR bekliyor
  chars_per_page INTEGER,
  meta        TEXT,                -- json: producer, year, ...
  collection  TEXT NOT NULL DEFAULT 'textbook',  -- 'textbook' | 'notes'
  lang        TEXT NOT NULL DEFAULT 'en'
);

CREATE INDEX IF NOT EXISTS books_collection ON books(collection);

CREATE TABLE IF NOT EXISTS pages(
  book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
  page    INTEGER NOT NULL,        -- 1-tabanli PDF sayfa numarasi
  text    TEXT,
  PRIMARY KEY(book_id, page)
);

CREATE TABLE IF NOT EXISTS toc(
  id      INTEGER PRIMARY KEY,
  book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
  level   INTEGER NOT NULL,
  title   TEXT NOT NULL,
  page    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS toc_book ON toc(book_id, page);

CREATE TABLE IF NOT EXISTS chunks(
  id         INTEGER PRIMARY KEY,
  book_id    INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
  page_start INTEGER NOT NULL,
  page_end   INTEGER NOT NULL,
  section    TEXT,                 -- "Bolum > Alt-bolum"
  text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_book ON chunks(book_id, page_start);

-- BM25 (leksik) katman
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,
  section,
  content='chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

-- Dense katman: float16 vektor, satir bazli
CREATE TABLE IF NOT EXISTS embeddings(
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  vec      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
