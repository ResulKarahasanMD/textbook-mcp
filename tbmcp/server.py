#!/usr/bin/env python3
"""textbook-mcp — yerel tip ders kitabi korpusu icin MCP sunucusu (stdio).

Ortam degiskenleri:
    TEXTBOOK_DB       corpus.sqlite yolu (zorunlu degil, varsayilan ./corpus.sqlite)
    TEXTBOOK_DENSE    0 ise yalniz BM25 (model yuklenmez, hizli ve hafif)
    TEXTBOOK_RERANK   0 ise cross-encoder devre disi
    TEXTBOOK_WARMUP   1 ise modeller+matris ACILISTA isitilir (varsayilan: tembel)
    TEXTBOOK_IDLE_UNLOAD  N saniye istek gelmezse modeller+matris bellekten
                      dusurulur (varsayilan 60; 0 = kapali). Surec YASAR,
                      stdio baglantisi kopmaz; bedel sonraki sorguda yeniden yukleme.

Tasarim: sunucu YALNIZ arama ve getirme yapar. Sentezi/yorumu cagiran model
(Claude) yapar — DynaMed MCP ile ayni bolusum.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

# stdio MCP'de stdout YALNIZ protokol icindir. Kutuphane loglarini stderr'e sabitle
# ve HF'yi cevrimdisi calistir (modeller zaten yerel; aksi halde her acilista ag).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.basicConfig(level=logging.WARNING, stream=__import__("sys").stderr)
for _n in ("sentence_transformers", "transformers", "httpx", "huggingface_hub", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import threading

from mcp.server.mcpserver import MCPServer, Image
from mcp.server.mcpserver.exceptions import ToolError

from . import search as S

# Varsayilan yol calisma dizinine degil pakete gore: sunucu hangi cwd'den
# baslatilirsa baslatilsin ayni korpusu bulur.
DB_PATH = os.environ.get("TEXTBOOK_DB") or str(
    Path(__file__).resolve().parent.parent / "corpus.sqlite")
USE_DENSE = os.environ.get("TEXTBOOK_DENSE", "1") != "0"
USE_RERANK = os.environ.get("TEXTBOOK_RERANK", "1") != "0"
# Isitma artik OPT-IN. stdio MCP sunuculari oturum basina ayri surec olarak
# dogar; kosulsuz isitma her oturumda matris (1.8 GB) + bge-m3 + reranker
# = ~8 GB'i, o oturum textbook'u hic kullanmasa bile ayiriyordu. 7 es zamanli
# oturumda bu 56 GB'a cikip 18 GB'lik makineyi 42 GB takasa gomdu (olculdu).
# Tembel tekil nesneler zaten ilk cagrida yukluyor; bedeli ilk aramada ~8 sn.
USE_WARMUP = os.environ.get("TEXTBOOK_WARMUP", "0") == "1"
# Bosta bosaltma: isitmanin simetrigi. Isitma HIC kullanilmayan oturumda bellek
# ayiriyordu; bu ise KULLANILMIS oturumda bellegin suresiz tutulmasini onler.
# Olculdu 2026-09-22, CANLI sunucu (Apple MPS): arama sonrasi RSS 2502 MB;
# bosaltmadan 60 sn sonra 926 MB ve orada SABIT (2 dk boyunca sapma yok).
# Oturum basina kazanc ~1.6 GB. Kalan 926 MB modeller DEGIL -- corpus_info
# "su_an_bellekte" ucunu de false gosteriyor; bu, Python/torch ayiricisinin
# 982.878 satirlik matristen kalan arenayi macOS'a geri vermemesi. Hic arama
# yapmamis oturum ~76 MB'ta kalir (tembel yukleme). 6 es zamanli oturumda
# takas 12 GB'a cikmisti. Bedel: sonraki aramada soguk yeniden yukleme
# (olculdu 11.9-16.3 sn; ilk yukleme 391+393 agirlik parcasi).
IDLE_UNLOAD = max(0, int(os.environ.get("TEXTBOOK_IDLE_UNLOAD", "60")))
_last_activity = time.monotonic()
_activity_lock = threading.Lock()


_inflight = 0


def _tracked(fn):
    """Arac cagrisini 'ucusta' say ve BITISTE saati sifirla.

    Sadece girise bakmak yetmiyordu: modelleri yukleyen ilk arama 20 sn suruyor
    ve o sure boyunca aktivite tazelenmedigi icin gozcu isin ORTASINDA bosaltma
    yapiyordu (olculdu, 5 sn'lik esikle yeniden uretildi). Sayac sifirdan buyukken
    gozcu hic dokunmaz; sure de cagri bitince baslar -- "sorgu bittikten N sn
    sonra" tanimi tam olarak bu. Ic ice cagri guvenli (find_in_book -> search).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _inflight, _last_activity
        with _activity_lock:
            _inflight += 1
        try:
            return fn(*args, **kwargs)
        finally:
            with _activity_lock:
                _inflight -= 1
                _last_activity = time.monotonic()
    return wrapper

mcp = MCPServer("textbook", instructions=__doc__)
_con: sqlite3.Connection | None = None
_db_lock = threading.Lock()


def db() -> sqlite3.Connection:
    global _con
    if _con is None:
        with _db_lock:
            if _con is None:
                p = Path(DB_PATH).expanduser()
                if not p.exists():
                    raise ToolError(
                        f"Korpus bulunamadi: {p}. Once `python -m tbmcp.ingest` calistir.")
                # SALT-OKUNUR: sunucu hicbir zaman yazmaz; ayni dosyaya es zamanli
                # yazan embed isiyle kilit yarisina girmesin.
                _con = sqlite3.connect(f"file:{p}?mode=ro", uri=True,
                                       check_same_thread=False)
                # mmap: VARSAYILAN KAPALI -- olculdu, kazanci yok.
                # 2026-09-22, 982.878 satirlik matris, ayri sureclerde 4'er
                # tekrar: acik ort. 2,85 s (medyan 2,08), kapali ort. 2,61 s
                # (medyan 2,32). Fark gurultunun icinde; her iki seride birer
                # aykiri deger (5,33 / 3,86 s) var. Kabul olcutu >=%10 idi.
                # DIKKAT: yalniz ISINMIS hal olculdu. SOGUK acilis (mmap'in
                # yardim etmesi beklenen tek durum) olculemedi: `purge` root
                # istiyor. Soguk olcmek icin `sudo purge` sonrasi:
                #   TEXTBOOK_MMAP=2147483648 .venv/bin/python -c "..."
                mmap_bytes = int(os.environ.get("TEXTBOOK_MMAP", 0))
                if mmap_bytes:
                    _con.execute(f"PRAGMA mmap_size={mmap_bytes}")
                # WAL'da okuyucu yazicidan engellenmez; bu yalniz checkpoint/
                # kurtarma anlarina karsi ucuz sigorta. Kazanc iddia edilmiyor.
                _con.execute("PRAGMA busy_timeout=5000")
    return _con


def _book_id(slug: str) -> tuple[int, str, int]:
    row = db().execute(
        "SELECT id,title,n_pages FROM books WHERE slug=?", (slug,)).fetchone()
    if not row:
        known = [r[0] for r in db().execute("SELECT slug FROM books LIMIT 10")]
        raise ToolError(f"Bilinmeyen kitap: {slug!r}. Ornekler: {known}")
    return row


# --------------------------------------------------------------------- araclar
@mcp.tool()
@_tracked
def textbook_list_books() -> str:
    """Korpustaki tum ders kitaplarini listele (slug, baslik, sayfa sayisi).

    Diger araclarin `book` parametresi buradaki `slug` degerini bekler.
    Bir soruyu yanitlamadan once korpusta hangi kaynaklarin oldugunu gormek icin kullan.
    """
    rows = db().execute(
        "SELECT b.slug,b.title,b.n_pages,COUNT(c.id),b.scanned,b.collection,b.lang"
        " FROM books b LEFT JOIN chunks c ON c.book_id=b.id"
        " GROUP BY b.id ORDER BY b.collection, b.title").fetchall()
    if not rows:
        return "Korpus bos."
    out = [f"{len(rows)} kitap:\n"]
    cur = None
    for slug, title, pages, n, scanned, coll, lang in rows:
        if coll != cur:
            cur = coll
            kac = sum(1 for r in rows if r[5] == coll)
            out.append(f"\n== koleksiyon: {coll} ({lang}) — {kac} kaynak ==")
        flag = "  [TARANMIS/OCR yok]" if scanned else ""
        out.append(f"- {slug}\n    {title}\n    {pages} sayfa, {n} parca{flag}")
    return "\n".join(out)


@mcp.tool()
@_tracked
def textbook_search(query: str, book: str = "", top_k: int = 8) -> str:
    """Ders kitaplarinda hibrit arama (BM25 + anlamsal + yeniden siralama).

    Klinik sorularin ilk adimi budur. Sorguyu kitaplarin diliyle (cogu korpusta
    Ingilizce tibbi terimlerle) yaz. Baska dilde sorgu yalniz anlamsal katman
    acikken calisir — once textbook_corpus_info ile dense_acik degerine bak;
    false ise sorgu kitaplarin dilinde olmak ZORUNDA.

    KAPSAM: yalniz 'textbook' koleksiyonu. Ders notlari ('notes' koleksiyonu)
    icin textbook_notes_search kullan — bu arac onlari DONDURMEZ.
    Kilit bilincli: olculdu, notlarin dilindeki bir sorguda not parcalari BM25
    uzerinden 30'luk aday havuzuna girip yuksek skorlu bir ders kitabi parcasini
    reranker'a hic ulastirmadan disari itebiliyordu. Iki koleksiyon ayri
    arandiginda ikisi de tam geri cagriliyor.

    Args:
        query: Aranacak ifade, orn. "diabetic ketoacidosis insulin infusion rate"
        book:  Tek bir kitapla sinirlamak icin slug (bos = tum ders kitabi korpusu)
        top_k: Dondurulecek parca sayisi (varsayilan 8)

    Doner: her biri [kitap s.N] kunyeli metin parcalari. Kunyeyi cevabinda
    aynen aktar; tam baglam icin textbook_get_page ile sayfayi ac.
    """
    hits = S.search(db(), query, top_k=max(1, min(top_k, 25)),
                    book=book or None, dense=USE_DENSE, use_rerank=USE_RERANK,
                    collection="textbook")
    return _fmt_hits(hits, query,
                     "Ders kitaplarinda bulunamadi; ders notlari icin "
                     "textbook_notes_search dene.")


def _fmt_hits(hits, query: str, bos_ipucu: str = "") -> str:
    """Arama sonuclarini kunyeli metne cevir (textbook_search ve textbook_notes_search ortak)."""
    if not hits:
        return (f"'{query}' icin korpusta sonuc yok. "
                + (bos_ipucu or "Terimi degistirip tekrar dene veya "
                                "textbook_list_books ile kapsami gor."))
    out = []
    for i, h in enumerate(hits, 1):
        head = f"{i}. {h.cite()}  skor={h.score:.3f}"
        if h.section:
            head += f"  |  {h.section[:80]}"
        out.append(f"{head}\n{h.text}")
    return "\n\n---\n\n".join(out)


@mcp.tool()
@_tracked
def textbook_notes_search(query: str, book: str = "", top_k: int = 8) -> str:
    """Ders notlarinda ('notes' koleksiyonu) arama yap.

    Ders notlarinin dilinde sorulan bir soru icin BUNU kullan; ders kitaplari
    icin textbook_search kullan. Ikisi AYRI kaynaklardir ve birbirinin sonucunu
    dondurmez; kapsamli bir yanit icin ikisini de cagirabilirsin.

    Neden ayri arac: olculdu — ders kitabi korpusu onlarca kat buyuk oldugu ve
    ayni-dil eslesmesi capraz-dil eslesmesinden yuksek skorladigi icin, tek
    havuzda arandiginda not parcalari kendi dillerindeki sorgularda bile yuzeye
    cikamiyor.

    Args:
        query: Notlarin dilinde ifade, orn. "diyabetik ketoasidoz insulin dozu"
        book:  Tek bir not dosyasiyla sinirlamak icin slug (textbook_list_books ile gor)
        top_k: Dondurulecek parca sayisi (varsayilan 8)

    Doner: [not s.N] kunyeli metin parcalari. Tablolar metin katmaninda
    bozulabilir — sayilar sutun sutun bozuk gorunuyorsa ayni sayfa icin
    textbook_page_image cagir.
    """
    hits = S.search(db(), query, top_k=max(1, min(top_k, 25)),
                    book=book or None, dense=USE_DENSE, use_rerank=USE_RERANK,
                    collection="notes")
    return _fmt_hits(hits, query,
                     "Ders notlarinda bulunamadi; ders kitaplari icin "
                     "textbook_search dene.")


@mcp.tool()
@_tracked
def textbook_get_page(book: str, page: int, window: int = 1) -> str:
    """Bir kitabin belirli sayfasini (ve komsularini) tam metin olarak getir.

    textbook_search bir parca dondurdukten sonra, kesilmis baglami tamamlamak
    icin kullan. window=1 onceki ve sonraki sayfayi da getirir.

    Args:
        book:   kitap slug'i
        page:   PDF sayfa numarasi (arama sonuclarindaki s.N degeri)
        window: kac komsu sayfa (0-5)
    """
    bid, title, n_pages = _book_id(book)
    if not 1 <= page <= n_pages:
        raise ToolError(f"{book} {n_pages} sayfa; s.{page} yok.")
    window = max(0, min(window, 5))
    lo, hi = max(1, page - window), min(n_pages, page + window)
    rows = db().execute(
        "SELECT page,text FROM pages WHERE book_id=? AND page BETWEEN ? AND ?"
        " ORDER BY page", (bid, lo, hi)).fetchall()
    if not rows:
        return f"{book} s.{page} bos veya yok (kitap {n_pages} sayfa)."
    parts = [f"# {title}"]
    total = 0
    for p, txt in rows:
        seg = f"\n----- [{book} s.{p}] -----\n{txt or '(bos sayfa)'}"
        total += len(seg)
        parts.append(seg)
        if total > 60_000:        # ~15k token tavani: 11 dev sayfa baglami sismesin
            parts.append(f"\n[kirpildi — kalan sayfalar icin window'u kucult]")
            break
    return "\n".join(parts)


@mcp.tool()
@_tracked
def textbook_find_in_book(book: str, query: str, top_k: int = 10) -> str:
    """Tek bir kitabin icinde ara. Bir kaynagin belirli bir konuda ne dedigini
    ogrenmek veya kitaplari karsilastirmak icin kullan.
    """
    return textbook_search(query=query, book=book, top_k=top_k)


@mcp.tool()
@_tracked
def textbook_toc(book: str, depth: int = 2, contains: str = "") -> str:
    """Bir kitabin icindekiler agacini getir (PDF yer imlerinden).

    Args:
        book:     kitap slug'i
        depth:    kac seviye derine inilsin (1-6)
        contains: yalniz bu ifadeyi iceren basliklar (bos = hepsi)
    """
    bid, title, _ = _book_id(book)
    sql = "SELECT level,title,page FROM toc WHERE book_id=? AND level<=?"
    params: list = [bid, max(1, min(depth, 6))]
    if contains:
        sql += " AND title LIKE ?"
        params.append(f"%{contains}%")
    sql += " ORDER BY page,level LIMIT 800"
    rows = db().execute(sql, params).fetchall()
    if not rows:
        return f"{book}: PDF yer imi yok. textbook_search ile ara."
    return f"# {title}\n" + "\n".join(
        f"{'  ' * (lvl - 1)}{t.strip()}  ....  s.{pg}" for lvl, t, pg in rows)


@mcp.tool()
@_tracked
def textbook_page_image(book: str, page: int, dpi: int = 150) -> Image:
    """Bir sayfayi GORUNTU olarak getir. Tablo, akis semasi, algoritma ve sekiller
    icin kullan — bunlar duz metne cevrildiginde bozulur.

    Args:
        book: kitap slug'i
        page: PDF sayfa numarasi
        dpi:  cozunurluk (72-200)
    """
    import pymupdf
    row = db().execute("SELECT path, n_pages, sha256 FROM books WHERE slug=?",
                       (book,)).fetchone()
    if not row:
        raise ToolError(f"Bilinmeyen kitap: {book!r}")
    path, n_pages, sha = row
    if not os.path.isabs(path):
        # Goreli yol sunucunun cwd'sine gore cozulurdu ve cwd kasitli olarak
        # belirsiz (bkz. DB_PATH notu) — sessizce yanlis dosya acmaktansa durur.
        raise ToolError(f"{book}: books.path goreli ({path!r}); mutlak yola cevrilmeli.")
    try:
        doc = pymupdf.open(path)
    except Exception:
        raise ToolError(f"PDF acilamadi (tasinmis olabilir): {path}")
    with doc:
        # Yol ayni kalip dosya BASKA bir baskiyla degistirilirse sayfa numaralari
        # kayar ve arac sessizce yanlis sayfayi gosterirdi. Sayfa sayisi bedava
        # bir kimlik kontrolu (dokuman zaten acik); farkli baski neredeyse her
        # zaman farkli sayfa sayisi tasir. Kesin dogrulama icin books.sha256 var.
        if n_pages and doc.page_count != n_pages:
            raise ToolError(
                f"{book}: PDF {doc.page_count} sayfa, korpus {n_pages} bekliyor — "
                f"dosya degismis olabilir (farkli baski?). Dogrula: "
                f"shasum -a 256 {path!r}  ->  beklenen {sha or '(kayitli degil)'}")
        if not 1 <= page <= doc.page_count:
            raise ToolError(f"{book} {doc.page_count} sayfa; s.{page} yok.")
        # Uzun kenari 1568 px ile sinirla: API zaten 1568 uzerini kucultuyor,
        # fazlasi modele ulasmayan bant genisligi. Buyuk formatli sayfa 200 dpi'da
        # 6+ MB PNG uretiyordu; tavan + JPEG fallback en kotuyu ~1 MB altina indirir.
        rect = doc[page - 1].rect
        zoom = min(max(72, min(dpi, 200)) / 72,
                   1568 / max(rect.width, rect.height))
        pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        png = pix.tobytes("png")
        if len(png) > 800_000:
            return Image(data=pix.tobytes("jpg", jpg_quality=80), format="jpeg")
        return Image(data=png, format="png")


@mcp.tool()
@_tracked
def textbook_corpus_info() -> str:
    """Korpusun durumu: kitap/parca sayisi, vektor katmani hazir mi, hangi model."""
    c = db()
    n_books, n_chunks, n_pages, n_vec = (
        c.execute("SELECT COUNT(*) FROM books").fetchone()[0],
        c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        c.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
        c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    meta = dict(c.execute("SELECT k,v FROM meta").fetchall())
    koll = {
        r[0]: {"kitap": r[1], "sayfa": r[2], "parca": r[3], "dil": r[4],
               "arac": ("textbook_notes_search" if r[0] == "notes"
                        else "textbook_search")}
        for r in c.execute(
            "SELECT b.collection,COUNT(DISTINCT b.id),SUM(b.n_pages),"
            "(SELECT COUNT(*) FROM chunks k JOIN books b2 ON b2.id=k.book_id"
            "  WHERE b2.collection=b.collection),MIN(b.lang)"
            " FROM books b GROUP BY b.collection")}
    return json.dumps({
        "db": str(Path(DB_PATH).expanduser()),
        "kitap": n_books, "sayfa": n_pages, "parca": n_chunks,
        "koleksiyonlar": koll,
        "vektor": n_vec,
        "vektor_kapsami": f"{100*n_vec/n_chunks:.1f}%" if n_chunks else "0%",
        "dense_acik": USE_DENSE, "rerank_acik": USE_RERANK,
        "bosta_bosaltma_sn": IDLE_UNLOAD,
        "ucustaki_istek": _inflight,
        "su_an_bellekte": S.loaded(),
        **meta,
    }, ensure_ascii=False, indent=2)


def _warmup() -> None:
    """Modelleri ve matrisi el sikismasindan sonra ARKA PLANDA isit.

    Eskiden ilk textbook_search cagrisi ~8 sn model yuklemesini icinde tasiyordu
    (Metal shader derlemesi dahil). Kilitler (search._model_lock/_mat_lock) ayni
    anda gelen ilk sorguyla cifte yuklemeyi onler. Isitmaya minik bir encode +
    rerank cifti dahil: sadece nesneyi kurmak shader derlemesini odemez.
    """
    try:
        con = db()
        if USE_DENSE:
            from .embed import encode, get_device
            S._fresh_matrix(con)
            encode(["warmup"], get_device())
        if USE_RERANK:
            S.rerank("warmup", [S.Hit(0, "w", "w", 1, 1, "", "warmup text")], 1)
    except Exception as e:                      # isitma hatasi olumcul degil
        logging.warning("isitma atlandi: %s", e)


def _idle_watchdog() -> None:
    """IDLE_UNLOAD sn istek gelmezse modelleri/matrisi dusur, sonra beklemeye don.

    Surec OLMEZ: stdio borusu acik kalir, istemci yeniden baglanmak zorunda
    degil. Uyanma araligi kalan sureye gore; bosaltma sonrasi 30 sn'de bir
    sessizce dolanir (unload() bos donunce maliyet yok).
    """
    while True:
        with _activity_lock:
            idle = time.monotonic() - _last_activity
            busy = _inflight
        if busy == 0 and idle >= IDLE_UNLOAD:
            try:
                if S.unload():
                    logging.warning("bosta %.0f sn -> modeller bellekten dusuruldu", idle)
            except Exception as e:                  # bosaltma hatasi olumcul degil
                logging.warning("bosta bosaltma basarisiz: %s", e)
            time.sleep(min(IDLE_UNLOAD, 30))
        else:
            time.sleep(max(1.0, IDLE_UNLOAD - idle))


def main() -> None:
    if USE_WARMUP:
        threading.Thread(target=_warmup, daemon=True).start()
    if IDLE_UNLOAD > 0:
        threading.Thread(target=_idle_watchdog, daemon=True).start()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
