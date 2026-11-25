#!/usr/bin/env python3
"""Embed the chunks (bge-m3, local, MPS) and write them into corpus.sqlite.

    python -m tbmcp.embed --db corpus.sqlite            # process the missing ones
    python -m tbmcp.embed --db corpus.sqlite --batch 24

bge-m3 is multilingual: a question asked in Turkish can find English textbook text.
Vectors are stored as float16 (half the disk; the quality difference is not measurable).
"""
from __future__ import annotations

import argparse
import mmap
import os
import sqlite3
import sys
import time

import numpy as np

# The models are already local; otherwise every startup goes out to the Hub (see server.py).
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
    """Move a module loaded on the CPU (fp32) to the device as fp16; large tensors piece by piece.

    The old path (load fp32 on the device, then .half()) left a 2773 MB footprint per
    568 M-parameter model: the MPS allocator places the fp16 tensors into the vacated fp32
    buffers (1682 MiB live, versus the expected 1083) and does not return the 977 MiB fp32
    embedding buffer to the driver even on empty_cache. The reverse does not work either:
    converting the whole model to fp16 on the CPU and then moving it leaves the same memory
    on the CPU side (2734 MB). Moving tensor by tensor, fp32 never reaches the device:
    1683 MB per model (2026-09-30, measured with footprint; encode output identical to 5
    decimal places). The fp32 -> fp16 conversion happens in a single CPU staging buffer of
    `block` elements (16 MB); with a CPU temporary per tensor, the freed blocks stayed in
    the allocator and remained in the footprint even after unloading.
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
                if buf is not None:             # position/type ids are integers: dtype is kept
                    mod._buffers[name] = (buf.to(device, dtype=torch.float16)
                                          if buf.is_floating_point() else buf.to(device))
    module.to(device)                           # tensors already in place; only the module's device info
    if device == "mps":
        torch.mps.empty_cache()


def load_model(device: str, model_name: str = MODEL):
    global _model
    with _model_lock:
      if _model is None:
        from sentence_transformers import SentenceTransformer
        gpu = device in ("mps", "cuda")
        m = SentenceTransformer(model_name, device="cpu" if gpu else device)
        # Our chunks are ~300 tokens; 512 instead of 1024 + fp16 is ~37% faster on MPS
        # (measured: 13.3 -> 18.3 chunks/s). The quality difference is negligible for
        # normalized cosine; consistent, since the query side is encoded with the same model.
        m.max_seq_length = 512
        if gpu:
            move_fp16(m, device)
        _model = m
    return _model


def unload_model() -> bool:
    """Drop the model from memory; the next load_model call loads it again.

    For idle unloading (see server._idle_watchdog). encode() holds the model in a local
    variable, so a call in progress is not affected.
    Returns: True if something was actually unloaded.
    """
    global _model
    with _model_lock:
        if _model is None:
            return False
        _model = None
        return True


def _fast_exit(code: int) -> None:
    """Exit the process without going through finalization.

    A normal CPython shutdown hung for minutes under this workload (measured: 12+ min,
    1% CPU). Three causes act together:
      1. multiprocessing.resource_tracker.__del__ blocks in waitpid with no timeout on
         macOS + py3.12 (cpython#88887). PyTorch has an atexit workaround, but atexit
         only runs after threading._shutdown() returns, and ~34 threads (OpenMP
         intra-op) remain in the process.
      2. Under swap, GC-ing hundreds of thousands of chunk strs is page-fault bound.
      3. Tearing down the model + the fp16 matrix.
    All writes must be committed before the call; since os._exit skips finalization
    entirely, it neutralizes all three at once.
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
    """An (n, dim) float16 array on anonymous mmap; the memory returns to the OS when the last reference goes.

    A 2.4 GB matrix allocated with np.empty stayed in the footprint even after deletion:
    macOS libmalloc keeps freed large blocks in its "large cache" (measured: a deleted
    2.3 GB showed as 2734 MB in `footprint`; the same block allocated with mmap dropped to
    408 MB). Only because of this does idle unloading (server._idle_watchdog) actually
    return memory.
    """
    if n * dim == 0:
        return np.zeros((n, dim), dtype=np.float16)
    return np.frombuffer(mmap.mmap(-1, n * dim * 2), dtype=np.float16).reshape(n, dim)


def load_matrix(con: sqlite3.Connection, block: int = 8192
                ) -> tuple[np.ndarray, np.ndarray]:
    """Load all vectors into memory as a single matrix. (ids, matrix[n,dim] float16)

    The old version used fetchall() + b"".join; both are full-size, so at the peak
    TWO copies of the matrix sat in RAM (1.84 -> 3.68 GB at 963,868 vectors).
    Here the target array is preallocated and filled from the cursor in blocks:
    peak = one copy + a temporary buffer of `block` rows (8192 x 2 KB = 16 MB).

    dim is derived from the first row; no assumption is made, so that the DIM constant
    cannot silently produce a wrong reshape after switching to a different model.
    """
    n, = con.execute("SELECT count(*) FROM embeddings").fetchone()
    if not n:
        return np.zeros(0, dtype=np.int64), np.zeros((0, DIM), dtype=np.float16)

    cur = con.execute("SELECT chunk_id, vec FROM embeddings ORDER BY chunk_id")
    first = cur.fetchmany(block)
    dim = len(first[0][1]) // 2                 # float16 -> 2 bytes
    ids = np.empty(n, dtype=np.int64)
    mat = empty_f16(n, dim)

    i, rows = 0, first
    while rows:
        k = min(len(rows), n - i)               # overflow if the embed job adds rows meanwhile
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
    ap.add_argument("--limit", type=int, default=0, help="for testing: the first N chunks")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    todo = con.execute(
        "SELECT c.id, c.text FROM chunks c"
        " LEFT JOIN embeddings e ON e.chunk_id = c.id"
        " WHERE e.chunk_id IS NULL ORDER BY c.id"
        + (f" LIMIT {args.limit}" if args.limit else "")
    ).fetchall()
    if not todo:
        print("no missing vectors.")
        con.close()
        return 0

    device = get_device(args.device)
    print(f"{len(todo)} chunks, model={args.model}, device={device}", flush=True)
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
        print(f"  {done}/{len(todo)}  {rate:.1f} chunks/s  ~{eta:.0f} min left",
              file=sys.stderr, flush=True)

    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('embed_model',?)",
                (args.model,))
    con.commit()
    # Closing checkpoints the WAL; better to pay for it here, at a measurable point,
    # than during unpredictable finalization.
    con.close()
    print(f"done: {done} vectors, {time.time()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    _fast_exit(main())
