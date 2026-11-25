#!/usr/bin/env python3
"""Parcalari vektorlestir (bge-m3, yerel, MPS) ve corpus.sqlite icine yaz.

    python -m tbmcp.embed --db corpus.sqlite            # eksik olanlari isle
    python -m tbmcp.embed --db corpus.sqlite --batch 24

bge-m3 cok dilli: Turkce sorulan soru Ingilizce kitap metnini bulabilir.
Vektorler float16 olarak saklanir (yarim disk, kalite farki olculemez seviyede).
"""
from __future__ import annotations

import argparse
import mmap
import os
import sqlite3
import sys
import time

import numpy as np

# Modeller zaten yerel; aksi halde her acilista Hub'a cikiyor (bkz. server.py).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "BAAI/bge-m3"
DIM = 1024


def get_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


_model = None
_model_lock = __import__("threading").Lock()


def move_fp16(module, device: str, block: int = 8_000_000) -> None:
    """CPU'da (fp32) yuklenmis modulu cihaza fp16 olarak tasi; buyuk tensorleri parca parca.

    Eski yol (cihazda fp32 yukle, sonra .half()) 568 M parametrelik model basina 2773 MB
    ayak izi birakiyordu: MPS ayiricisi fp16 tensorleri bosalan fp32 tamponlarina
    yerlestiriyor (canli 1682 MiB, olmasi gereken 1083) ve 977 MiB'lik fp32 gomme
    tamponunu empty_cache'te bile surucuye geri vermiyor. Tersi de olmuyor: tum modeli
    CPU'da fp16'ya cevirip tasimak ayni bellegi CPU tarafinda birakiyor (2734 MB).
    Tensor tensor tasiyinca fp32 cihaza hic cikmiyor: model basina 1683 MB (2026-09-30,
    footprint ile olculdu; encode ciktisi 5 ondalikta ayni). fp32 -> fp16 donusumu tek bir
    `block` elemanlik (16 MB) CPU hazirlama tamponunda yapilir; tensor basina CPU gecicisi
    ayrilsaydi serbest kalan bloklar ayiricida kalip bosaltmadan sonra da ayak izinde
    duruyordu.
    """
    import torch
    with torch.no_grad():
        staging = torch.empty(block, dtype=torch.float16)
        for p in module.parameters():
            src = p.data
            dst = torch.empty(src.shape, dtype=torch.float16, device=device)
            flat, out = src.reshape(-1), dst.view(-1)
            for i in range(0, flat.numel(), block):
                k = min(block, flat.numel() - i)
                staging[:k].copy_(flat[i:i + k])
                out[i:i + k].copy_(staging[:k])
            p.data = dst
        del staging
        for mod in module.modules():
            for name, buf in list(mod._buffers.items()):
                if buf is not None:             # konum/tip id'leri tamsayi: dtype korunur
                    mod._buffers[name] = (buf.to(device, dtype=torch.float16)
                                          if buf.is_floating_point() else buf.to(device))
    module.to(device)                           # tensorler yerinde; yalniz modulun cihaz bilgisi
    if device == "mps":
        torch.mps.empty_cache()


def load_model(device: str, model_name: str = MODEL):
    global _model
    with _model_lock:
      if _model is None:
        from sentence_transformers import SentenceTransformer
        gpu = device in ("mps", "cuda")
        m = SentenceTransformer(model_name, device="cpu" if gpu else device)
        # Parcalarimiz ~300 token; 1024 yerine 512 + fp16 MPS'te ~%37 hizlandiriyor
        # (olculdu: 13.3 -> 18.3 parca/sn). Kalite farki normalize kosinusde ihmal
        # edilebilir; sorgu tarafi da ayni modelle kodlandigi icin tutarli.
        m.max_seq_length = 512
        if gpu:
            move_fp16(m, device)
        _model = m
    return _model


def unload_model() -> bool:
    """Modeli bellekten dusur; sonraki load_model cagrisi yeniden yukler.

    Bosta bosaltma icin (bkz. server._idle_watchdog). encode() modeli yerel bir
    degiskende tuttugu icin devam eden bir cagri bundan etkilenmez.
    Donus: gercekten bir sey bosaltildiysa True.
    """
    global _model
    with _model_lock:
        if _model is None:
            return False
        _model = None
        return True


def _fast_exit(code: int) -> None:
    """Sureci finalization'a girmeden kapat.

    Normal CPython kapanisi bu is yukunde dakikalarca asiliyordu (olculdu: 12+ dk,
    %1 CPU). Uc sebep birlikte calisiyor:
      1. multiprocessing.resource_tracker.__del__ macOS + py3.12'de waitpid'de
         zaman asimisiz bloke olur (cpython#88887). PyTorch'un atexit cozumu var
         ama atexit yalniz threading._shutdown() dondukten sonra kosuyor ve
         surecte ~34 thread (OpenMP intra-op) duruyor.
      2. Takas altinda yuz binlerce chunk str'ini GC etmek sayfa-hatasi bagli.
      3. Model + fp16 matrisin yikimi.
    Tum yazmalar cagri oncesinde commit edilmis olmali; os._exit finalization'i
    tamamen atladigi icin ucunu birden etkisizlestirir.
    """
    unload_model()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def encode(texts: list[str], device: str, batch: int = 32,
           model_name: str = MODEL) -> np.ndarray:
    m = load_model(device, model_name)
    v = m.encode(texts, batch_size=batch, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False)
    return v.astype(np.float16)


def empty_f16(n: int, dim: int) -> np.ndarray:
    """Anonim mmap uzerinde (n, dim) float16 dizi; son referans gidince bellek OS'a doner.

    np.empty ile ayrilan 2,4 GB'lik matris silindikten sonra da ayak izinde kaliyordu:
    macOS libmalloc serbest kalan buyuk bloklari "large cache"te tutuyor (olculdu:
    silinmis 2,3 GB `footprint`te 2734 MB; ayni blok mmap ile ayrilinca 408 MB'a dustu).
    Bosta bosaltma (server._idle_watchdog) ancak bu sayede gercekten bellek geri verir.
    """
    if n * dim == 0:
        return np.zeros((n, dim), dtype=np.float16)
    return np.frombuffer(mmap.mmap(-1, n * dim * 2), dtype=np.float16).reshape(n, dim)


def load_matrix(con: sqlite3.Connection, block: int = 8192
                ) -> tuple[np.ndarray, np.ndarray]:
    """Tum vektorleri tek matris olarak bellege al. (ids, matrix[n,dim] float16)

    Eski hali fetchall() + b"".join kullaniyordu; ikisi de tam boyda, yani zirvede
    matrisin IKI kopyasi RAM'de duruyordu (963.868 vektorde 1.84 -> 3.68 GB).
    Burada hedef dizi onceden ayrilip imlecten bloklar halinde dolduruluyor:
    zirve = tek kopya + `block` satirlik gecici tampon (8192 x 2 KB = 16 MB).

    dim ilk satirdan cikarilir; DIM sabiti farkli bir modele gecildiginde
    sessizce yanlis reshape uretmesin diye varsayim yapilmaz.
    """
    n, = con.execute("SELECT count(*) FROM embeddings").fetchone()
    if not n:
        return np.zeros(0, dtype=np.int64), np.zeros((0, DIM), dtype=np.float16)

    cur = con.execute("SELECT chunk_id, vec FROM embeddings ORDER BY chunk_id")
    first = cur.fetchmany(block)
    dim = len(first[0][1]) // 2                 # float16 -> 2 bayt
    ids = np.empty(n, dtype=np.int64)
    mat = empty_f16(n, dim)

    i, rows = 0, first
    while rows:
        k = min(len(rows), n - i)               # embed isi arada satir eklerse tasma
        if k <= 0:
            break
        ids[i:i + k] = [r[0] for r in rows[:k]]
        mat[i:i + k] = np.frombuffer(
            b"".join(r[1] for r in rows[:k]), dtype=np.float16).reshape(k, dim)
        i += k
        rows = cur.fetchmany(block)
    cur.close()
    return (ids, mat) if i == n else (ids[:i], mat[:i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="corpus.sqlite")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="test icin: ilk N parca")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    todo = con.execute(
        "SELECT c.id, c.text FROM chunks c"
        " LEFT JOIN embeddings e ON e.chunk_id = c.id"
        " WHERE e.chunk_id IS NULL ORDER BY c.id"
        + (f" LIMIT {args.limit}" if args.limit else "")
    ).fetchall()
    if not todo:
        print("eksik vektor yok.")
        con.close()
        return 0

    device = get_device(args.device)
    print(f"{len(todo)} parca, model={args.model}, cihaz={device}", flush=True)
    t0 = time.time()
    done = 0
    STEP = args.batch * 16
    for i in range(0, len(todo), STEP):
        block = todo[i:i + STEP]
        vecs = encode([t for _, t in block], device, args.batch, args.model)
        con.executemany(
            "INSERT OR REPLACE INTO embeddings(chunk_id,vec) VALUES(?,?)",
            [(cid, v.tobytes()) for (cid, _), v in zip(block, vecs)])
        con.commit()
        done += len(block)
        dt = time.time() - t0
        rate = done / dt
        eta = (len(todo) - done) / rate / 60
        print(f"  {done}/{len(todo)}  {rate:.1f} parca/sn  kalan ~{eta:.0f} dk",
              file=sys.stderr, flush=True)

    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('embed_model',?)",
                (args.model,))
    con.commit()
    # Kapanis WAL'i checkpoint eder; belirsiz finalization yerine burada, olculebilir
    # bir noktada odenmesi tercih edilir.
    con.close()
    print(f"bitti: {done} vektor, {time.time()-t0:.0f} sn")
    return 0


if __name__ == "__main__":
    _fast_exit(main())
