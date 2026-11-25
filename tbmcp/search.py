#!/usr/bin/env python3
"""Hibrit arama: BM25 (FTS5) ∪ dense (bge-m3) -> RRF -> cross-encoder rerank.

CLI olarak da calisir:
    python -m tbmcp.search --db corpus.sqlite "diyabetik ketoasidoz tedavisi"
"""
from __future__ import annotations

import argparse
import gc
import re
import sqlite3
import sys
import threading
from dataclasses import dataclass, field

import numpy as np

RERANKER = "BAAI/bge-reranker-v2-m3"
RRF_K = 60  # Cormack ve ark. (2009) varsayilani

# (ids, mat, max_chunk_id) — max_chunk_id tazelik probu icin (bkz. _fresh_matrix)
_matrix: tuple[np.ndarray, np.ndarray, int] | None = None
# koleksiyon -> bool maske; _matrix ile ayni max_chunk_id probunda gecersizlesir
_coll_masks: dict[str, np.ndarray] = {}
_coll_masks_max: int = -1
# Filtreli aramada (tek kitap / kucuk koleksiyon) tum matris yerine yalniz o satirlar
# bellege alinir (bkz. _subset_matrix). (book, collection) -> (ids, mat, max_chunk_id);
# en son kullanilan _SUBSET_KEEP tanesi tutulur.
SUBSET_MAX_ROWS = 200_000
_SUBSET_KEEP = 2
SCORE_BLOCK_ROWS = 2048                 # puanlamadaki fp32 calisma tamponu (bkz. dense_search)
_subsets: dict[tuple[str | None, str | None], tuple[np.ndarray, np.ndarray, int]] = {}
# SUBSET_MAX_ROWS'u asan filtreler -> karar verildigi andaki max_chunk_id (her sorguda saymamak icin)
_subset_big: dict[tuple[str | None, str | None], int] = {}
_reranker = None
# Tembel tekil nesneler paralel MCP arac cagrilarinda kilitsiz yariisirdi;
# ayrica torch/MPS modelleri es zamanli cok-thread'li cikarim icin belgelenmis
# guvenli degil — _model_lock hem init'i hem encode/predict cagrilarini serilestirir.
_mat_lock = threading.Lock()
_model_lock = threading.Lock()


def loaded() -> dict[str, bool]:
    """Su an bellekte ne var. Bosta bosaltmayi dogrulamak icin corpus_info kullanir."""
    from . import embed
    return {"model": embed._model is not None,
            "reranker": _reranker is not None,
            "matris": _matrix is not None or bool(_subsets)}


def unload() -> bool:
    """Model, reranker ve vektor matrisini bellekten dusur.

    Kilit sirasi dense_search ile AYNI: once _model_lock (icinde embed'inki),
    sonra _mat_lock. Ikisi hicbir zaman ic ice alinmiyor -> kilitlenme yok.
    Devam eden bir arama ids/mat icin kendi yerel referansini tuttugundan
    bosaltma onu bozmaz; bellek is bitince serbest kalir.
    Donus: gercekten bir sey bosaltildiysa True.
    """
    global _matrix, _coll_masks, _coll_masks_max, _reranker
    freed = False
    with _model_lock:
        if _reranker is not None:
            _reranker = None
            freed = True
        from .embed import unload_model
        if unload_model():
            freed = True
    with _mat_lock:
        if _matrix is not None:
            _matrix = None
            freed = True
        if _coll_masks:
            _coll_masks = {}
            _coll_masks_max = -1
            freed = True
        if _subsets:
            _subsets.clear()
            freed = True
        _subset_big.clear()
    if freed:
        gc.collect()
        # torch onbellegi ayri tutulur, gc gormez. Yuklu degilse ice aktarma: bosaltma icin
        # torch'u yuklemek sureci ~120 MB buyutuyordu (olculdu, yalniz alt kume kullanan surecte).
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    return freed


@dataclass
class Hit:
    chunk_id: int
    book: str
    title: str
    page_start: int
    page_end: int
    section: str
    text: str
    n_pages: int = 0          # kitabin toplam sayfasi; dizin/arka-matter tespiti icin
    score: float = 0.0
    why: list[str] = field(default_factory=list)  # hangi katman buldu

    def cite(self) -> str:
        pg = (f"s.{self.page_start}" if self.page_start == self.page_end
              else f"s.{self.page_start}-{self.page_end}")
        return f"[{self.book} {pg}]"


# ------------------------------------------------------------------------ BM25
# FTS5 unicode61 harf+rakami TEK token sayar ("hba1c", "b12", "cd4", "t3").
# Sorgu tokenizasyonu ayni siniri kullanmali: eski hali harf/rakam dallarini
# ayirip HbA1c -> '"HbA" OR "1"' uretiyordu (olculdu: 0 gercek eslesme,
# 191k parca gurultu). Dallar: gercek ondalik ("7.35"), harf+rakam kosugu, tek sayi.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)+|[^\W_]{2,}|\d+", re.UNICODE)


def fts_query(q: str) -> str:
    """Serbest metni guvenli bir FTS5 MATCH ifadesine cevir (terimlerin OR'u)."""
    terms = _TOKEN.findall(q)
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms[:32])


def bm25_search(con: sqlite3.Connection, query: str, limit: int,
                book: str | None = None,
                collection: str | None = None) -> list[int]:
    m = fts_query(query)
    if not m:
        return []
    sql = ("SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
           "JOIN books b ON b.id = c.book_id WHERE f.chunks_fts MATCH ?")
    params: list = [m]
    if book:
        sql += " AND b.slug = ?"
        params.append(book)
    if collection:
        # SQL pre-filter: aday havuzu daraldigi icin filtre BM25'i ucuzlatir
        # (olculdu: tum korpus 95 ms, tek kitap 61 ms)
        sql += " AND b.collection = ?"
        params.append(collection)
    sql += " ORDER BY bm25(chunks_fts, 1.0, 0.4) LIMIT ?"
    params.append(limit)
    return [r[0] for r in con.execute(sql, params)]


# ----------------------------------------------------------------------- dense
def _fresh_matrix(con: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray, int]:
    """Vektor matrisini getir; embed isi kosarken yeni vektorleri ARTIMLI ekle.

    Prob: SELECT MAX(chunk_id) PK uzerinden ~0.1 ms (COUNT(*) 90 ms olculdu).
    embed.py yalniz yeni chunk_id INSERT ettigi icin mevcut satirlar degismez;
    artimli cekme bu yuzden dogru. Eski davranis: onbellek hic tazelenmiyordu,
    sunucu embed isinin %X anlik goruntusuyle donuk kaliyordu.
    """
    global _matrix
    row = con.execute("SELECT MAX(chunk_id) FROM embeddings").fetchone()
    cur_max = int(row[0] or 0)
    if _matrix is not None and _matrix[2] == cur_max:
        return _matrix
    with _mat_lock:
        from .embed import load_matrix
        if _matrix is None or len(_matrix[0]) == 0:
            ids, mat = load_matrix(con)
            _matrix = (ids, mat, cur_max)
        elif _matrix[2] != cur_max:
            ids0, mat0, last = _matrix
            rows = con.execute(
                "SELECT chunk_id, vec FROM embeddings WHERE chunk_id > ?"
                " ORDER BY chunk_id", (last,)).fetchall()
            if rows:
                new_ids = np.array([r[0] for r in rows], dtype=ids0.dtype)
                new_mat = np.frombuffer(b"".join(r[1] for r in rows),
                                        dtype=np.float16).reshape(len(rows), mat0.shape[1])
                _matrix = (np.concatenate([ids0, new_ids]),
                           np.vstack([mat0, new_mat]), cur_max)
            else:
                _matrix = (ids0, mat0, cur_max)
    return _matrix


def _id_mask(con: sqlite3.Connection, ids: np.ndarray, sql: str,
             params: tuple) -> np.ndarray:
    """ids icinde SQL'in dondurdugu chunk id'leri True olan bool maske.

    Eski hali `np.fromiter((i in allowed for i in ids), ...)` idi: 983k satirda
    olculen 65.8 ms, saf Python dongusu. np.isin ayni sonucu 2.5 ms'de veriyor
    (26x) ve bu bedel her textbook_find_in_book cagrisinda odeniyordu.
    """
    allowed = np.fromiter((r[0] for r in con.execute(sql, params)),
                          dtype=ids.dtype)
    return np.isin(ids, allowed)


def _collection_mask(con: sqlite3.Connection, ids: np.ndarray, cur_max: int,
                     collection: str) -> np.ndarray:
    """Koleksiyon maskesini onbellekten ver; matris tazelendiyse yeniden kur."""
    global _coll_masks, _coll_masks_max
    with _mat_lock:
        if _coll_masks_max != cur_max:
            _coll_masks, _coll_masks_max = {}, cur_max
        m = _coll_masks.get(collection)
        if m is None or len(m) != len(ids):
            m = _id_mask(con, ids,
                         "SELECT c.id FROM chunks c JOIN books b ON b.id=c.book_id"
                         " WHERE b.collection=?", (collection,))
            _coll_masks[collection] = m
    return m


def _subset_matrix(con: sqlite3.Connection, book: str | None,
                   collection: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Filtreye uyan satirlarin vektorlerini getir: (ids, mat). Filtre buyukse None.

    Tum matris 1,24 M satirda 2,5 GB tutuyor; tek kitapla sinirli aramada bunun en cok
    51 648 satiri (106 MB) gerekiyor. Filtre SUBSET_MAX_ROWS'u asarsa (textbook
    koleksiyonu = korpusun %98'i) None doner ve cagiran tum matrise duser.
    Tazelik _fresh_matrix ile ayni probla: max_chunk_id degistiyse alt kume yeniden cekilir.
    """
    from .embed import empty_f16
    key = (book, collection)
    cur_max = int(con.execute("SELECT MAX(chunk_id) FROM embeddings").fetchone()[0] or 0)
    where = " AND ".join(w for w, v in (("b.slug=?", book), ("b.collection=?", collection)) if v)
    params = tuple(v for v in (book, collection) if v)
    with _mat_lock:
        hit = _subsets.get(key)
        if hit is not None and hit[2] == cur_max:
            _subsets[key] = _subsets.pop(key)           # en son kullanilan sona
            return hit[0], hit[1]
        if _subset_big.get(key) == cur_max:
            return None
        # Ust sinir chunks_book indeksinden; embeddings'e katilmadan sayilir.
        n_max, = con.execute("SELECT COUNT(*) FROM chunks c JOIN books b ON b.id=c.book_id"
                             " WHERE " + where, params).fetchone()
        if n_max > SUBSET_MAX_ROWS:
            _subset_big[key] = cur_max
            return None
        cur = con.execute("SELECT e.chunk_id, e.vec FROM embeddings e"
                          " JOIN chunks c ON c.id=e.chunk_id JOIN books b ON b.id=c.book_id"
                          " WHERE " + where + " ORDER BY e.chunk_id", params)
        rows = cur.fetchmany(8192)
        dim = len(rows[0][1]) // 2 if rows else 0       # float16 -> 2 bayt
        ids = np.empty(n_max, dtype=np.int64)
        mat = empty_f16(n_max, dim)                     # mmap: dusurulunce OS'a doner
        i = 0
        while rows:
            k = min(len(rows), n_max - i)               # embed isi arada satir eklerse tasma
            if k <= 0:
                break
            ids[i:i + k] = [r[0] for r in rows[:k]]
            mat[i:i + k] = np.frombuffer(
                b"".join(r[1] for r in rows[:k]), dtype=np.float16).reshape(k, dim)
            i += k
            rows = cur.fetchmany(8192)
        cur.close()
        ids, mat = ids[:i], mat[:i]
        _subsets.pop(key, None)
        _subsets[key] = (ids, mat, cur_max)
        while len(_subsets) > _SUBSET_KEEP:
            _subsets.pop(next(iter(_subsets)))
    return ids, mat


def dense_search(con: sqlite3.Connection, query: str, limit: int,
                 book: str | None = None,
                 collection: str | None = None) -> list[int]:
    from .embed import get_device, encode

    # Tum matris zaten bellekteyse (genel arama yapilmis) onu maskele; degilse filtreli
    # arama icin yalniz gereken satirlari cek.
    sub = None
    if (book or collection) and _matrix is None:
        sub = _subset_matrix(con, book, collection)
    if sub is not None:
        ids, mat = sub
    else:
        ids, mat, cur_max = _fresh_matrix(con)
    if len(ids) == 0:
        return []

    with _model_lock:
        qv = encode([query], get_device())[0].astype(np.float32)
    # fp16 matrisi bloklar halinde TEK bir fp32 tampona kopyalayip carp: tam-matris astype
    # sorgu basina ~2 GB gecici tahsis ediyordu (olculdu: 679 ms -> 75 ms, sonuclar
    # bit-bit ayni). 64k satirlik `astype` bloklari da her biri 268 MB'lik gecici
    # birakiyordu ve serbest kalan buyuk bloklar bu makinede ayak izinden dusmuyor
    # (olculdu: silinen 762 MB MALLOC_LARGE'da kaldi). 2048 satirlik sabit tampon 8 MB;
    # 300k satirda 31,4 -> 21,6 ms, sonuc bit-bit ayni. Vektorler normalize, nokta = kosinus.
    scores = np.empty(len(ids), np.float32)
    buf = np.empty((min(SCORE_BLOCK_ROWS, len(ids)), mat.shape[1]), np.float32)
    for i in range(0, len(ids), SCORE_BLOCK_ROWS):
        k = min(SCORE_BLOCK_ROWS, len(ids) - i)
        np.copyto(buf[:k], mat[i:i + k])
        np.dot(buf[:k], qv, out=scores[i:i + k])

    mask = None
    if sub is None and collection:
        mask = _collection_mask(con, ids, cur_max, collection)
    if sub is None and book:
        m = _id_mask(con, ids,
                     "SELECT c.id FROM chunks c JOIN books b ON b.id=c.book_id"
                     " WHERE b.slug=?", (book,))
        mask = m if mask is None else (mask & m)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    n = min(limit, len(ids))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    return [int(ids[i]) for i in top if np.isfinite(scores[i])]


# ------------------------------------------------------- kaynakca ayiklama
# Ders kitaplarinin bolum sonu kaynakcalari sorgu terimlerini yogun icerdigi icin
# BM25'i yaniltiyor (olculdu: "tumor lysis syndrome rasburicase" -> ilk iki isabet
# Nathan & Oski atif listesi). Klinik icerik degil, atif dokumu olduklarindan
# elenmez, geri plana atilir: baska aday yoksa yine de donerler.
_CITE_PAT = re.compile(
    r"\bet\s+al\b"                      # "Annemans L, Moeremans K, et al"
    r"|\b\d{1,4}:\d{1,4}\s*[-–]\s*\d{1,4}\b"   # "14:1203-1210" cilt:sayfa
    r"|\bdoi:"
    r"|\b(?:19|20)\d{2}\s*[.;]"         # "2003." yil sonu
    r"|^\s*\d{1,4}\.\s+[A-Z]",          # "108. Annemans" numarali liste
    re.I | re.M)


def is_reference_dump(text: str, min_hits: int = 4) -> bool:
    """Parca bir kaynakca/atif listesi mi? Yogunluga bakar, tek atifa takilmaz."""
    hits = len(_CITE_PAT.findall(text))
    if hits < min_hits:
        return False
    # 400 karakterde 1.5+ atif isareti -> icerik degil, atif dokumu
    return hits / max(1.0, len(text) / 400) >= 1.5


# Kitap sonu dizinleri de ayni sekilde BM25'i yaniltiyor: bastan sona sorgu
# terimi, araya serpistirilmis sayfa numaralari, sifir duzyazi. Kaynakca desenine
# uymadiklari icin _CITE_PAT bunlari kaciriyordu (olculdu: bir laboratuvar tibbi kitabinda
# "troponin reference interval myocardial infarction" sorgusunun ILK isabeti
# bozuk bir dizin sayfasiydi -- "Cardiac troponin (cTnJ assays {Continued}").
_INDEX_PAT = re.compile(
    r",\s*\d{1,4}[a-z]?(?:\s*[-–—]\s*\d{1,4}[a-z]?)?(?=\s*(?:[,;]|$))", re.M)
_INDEX_SEC = re.compile(r"^(?:subject\s+|author\s+|name\s+)?index$", re.I)


def is_index_dump(text: str, min_hits: int = 6) -> bool:
    """Parca kitap sonu dizini mi? 'Terim, 12, 15-17, 20t' deseninin yogunlugu."""
    hits = len(_INDEX_PAT.findall(text))
    if hits < min_hits:
        return False
    # 400 karakterde 2.5+ sayfa-referansi -> dizin. Esik kaynakcanınkinden (1.5)
    # yuksek: duzyazida ", 45" gibi tek eslesmeler gecebiliyor, yogunlugu gecemiyor.
    return hits / max(1.0, len(text) / 400) >= 2.5


def is_low_value(hit: "Hit") -> bool:
    """Klinik icerik degil: kaynakca dokumu ya da kitap sonu dizini.

    Elenmez, geri plana atilir -- baska aday yoksa yine de doner (bkz. ref-fill).

    Dizin icin yogunluk TEK BASINA yetmiyor: sayisal klinik tablolar da ayni
    desene giriyor (olculdu: 4000 parcalik orneklemde 97 isaret, 22'si kitap
    sonunda degildi ve biri gercek bir karotis stenoz kriter tablosuydu).
    Bu yuzden yogunluk + konum birlikte araniyor; dizin arka matterdadir.
    """
    leaf = (hit.section or "").split(">")[-1].strip()
    if leaf and _INDEX_SEC.match(leaf):
        return True
    if is_reference_dump(hit.text):
        return True
    if not is_index_dump(hit.text):
        return False
    if not hit.n_pages:                      # konum bilinmiyor -> yogunluga guven
        return True
    return hit.page_start / hit.n_pages > 0.85


# ------------------------------------------------------------------------- RRF
def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, list[str]]]:
    acc: dict[int, float] = {}
    src: dict[int, list[str]] = {}
    for name, ids in rankings.items():
        for rank, cid in enumerate(ids):
            acc[cid] = acc.get(cid, 0.0) + 1.0 / (k + rank + 1)
            src.setdefault(cid, []).append(name)
    return sorted(((cid, s, src[cid]) for cid, s in acc.items()),
                  key=lambda t: -t[1])


# -------------------------------------------------------------------- reranker
def rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    global _reranker
    if not hits:
        return hits
    from sentence_transformers import CrossEncoder
    from .embed import get_device, move_fp16
    with _model_lock:
        if _reranker is None:
            dev = get_device()
            gpu = dev in ("mps", "cuda")
            ce = CrossEncoder(RERANKER, device="cpu" if gpu else dev, max_length=512)
            if gpu:
                move_fp16(ce, dev)
            _reranker = ce
        scores = _reranker.predict([(query, h.text) for h in hits],
                                   batch_size=16, show_progress_bar=False)
    for h, s in zip(hits, scores):
        h.score = float(s)
        h.why = h.why + ["ce"]       # skorun cross-encoder'dan geldigini isaretle
    return sorted(hits, key=lambda h: -h.score)[:top_k]


# ------------------------------------------------------------------------ ana
def hydrate(con: sqlite3.Connection, scored: list[tuple[int, float, list[str]]]) -> list[Hit]:
    if not scored:
        return []
    order = {cid: i for i, (cid, _, _) in enumerate(scored)}
    q = ",".join("?" * len(order))
    rows = con.execute(
        f"SELECT c.id,b.slug,b.title,c.page_start,c.page_end,c.section,c.text,"
        f"b.n_pages FROM chunks c JOIN books b ON b.id=c.book_id WHERE c.id IN ({q})",
        list(order)).fetchall()
    hits = [Hit(r[0], r[1], r[2], r[3], r[4], r[5] or "", r[6], r[7] or 0)
            for r in rows]
    for h in hits:
        cid, s, why = scored[order[h.chunk_id]]
        h.score, h.why = s, why
    return sorted(hits, key=lambda h: order[h.chunk_id])


def release_device_cache() -> None:
    """Cikarim bitince MPS onbellegini surucuye geri ver.

    MPS ayiricisi aktivasyon tamponlarini cagri bittikten sonra da tutuyor; birlesik
    bellekte bu dogrudan RAM. Olculdu (2026-09-30): bes aramadan sonra geri verilebilir
    986 MB, cagri suresi 81 ms. torch hic yuklenmediyse (yalniz BM25) yuklemez.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    with _model_lock:                   # suren bir encode/predict ile yarisma
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def search(con: sqlite3.Connection, query: str, top_k: int = 8,
           book: str | None = None, candidates: int = 30,
           dense: bool = True, use_rerank: bool = True,
           collection: str | None = None) -> list[Hit]:
    try:
        return _search(con, query, top_k, book, candidates, dense, use_rerank, collection)
    finally:
        if dense or use_rerank:
            release_device_cache()


def _search(con: sqlite3.Connection, query: str, top_k: int,
            book: str | None, candidates: int,
            dense: bool, use_rerank: bool,
            collection: str | None) -> list[Hit]:
    rankings = {"bm25": bm25_search(con, query, candidates, book, collection)}
    if dense:
        try:
            d = dense_search(con, query, candidates, book, collection)
            if d:
                rankings["dense"] = d
        except Exception as e:                      # model yok / indirilmemis
            rankings.setdefault("_warn", [])
            print(f"[dense atlandi: {e}]", file=sys.stderr, flush=True)

    hits = hydrate(con, rrf({k: v for k, v in rankings.items() if v})[:candidates])
    # Birebir tekrarlari ele (ayni kitapta ayni metin — or. yinelenen sayfalar)
    seen_txt: set = set()
    hits = [h for h in hits
            if (key := (h.book, " ".join(h.text.split())[:200])) not in seen_txt
            and not seen_txt.add(key)]
    # Kaynakca ve dizin parcalarini sona at (elemek yerine: bazen tek kaynak odur)
    low = [is_low_value(h) for h in hits]
    real = [h for h, lo in zip(hits, low) if not lo]
    refs = [h for h, lo in zip(hits, low) if lo]
    hits = real + refs
    if use_rerank and len(hits) > top_k:
        try:
            # rerank YALNIZ gercek icerikte kosar; kaynakca aday yetmezse
            # rerank'ten SONRA arkaya eklenir — eski hali reranker'a karisik havuz
            # verip kaynakcayi yeniden one cikarabiliyordu.
            ranked = rerank(query, real, top_k) if real else []
            fill = refs[:max(0, top_k - len(ranked))]
            for h in fill:
                h.why = h.why + ["ref-fill"]   # RRF skorlu dolgu, CE skoru degil
            return ranked + fill
        except Exception as e:
            print(f"[rerank atlandi: {e}]", file=sys.stderr, flush=True)
    return hits[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--db", default="corpus.sqlite")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--book")
    ap.add_argument("--collection")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    import time
    t0 = time.time()
    hits = search(con, a.query, a.k, a.book, dense=not a.no_dense,
                  use_rerank=not a.no_rerank, collection=a.collection)
    print(f"({(time.time()-t0)*1000:.0f} ms)\n")
    for h in hits:
        print(f"{h.cite()} {h.score:+.3f} [{'+'.join(h.why)}] {h.section[:60]}")
        print("   " + " ".join(h.text.split())[:220] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
