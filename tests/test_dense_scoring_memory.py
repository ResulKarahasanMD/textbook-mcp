"""Scoring must not allocate an fp32 temporary array proportional to the matrix.

On this machine freed large numpy blocks stay in the footprint (measured: a deleted
762 MB remains in `footprint` as MALLOC_LARGE), so every temporary allocation is a lasting
cost. The test counts numpy allocations with tracemalloc.
"""
from __future__ import annotations

import os
import tracemalloc
import unittest

import numpy as np

from tbmcp import embed, search as S
from tests import fixture as F

DIM, SATIR = 64, 20_000


class PuanlamaBellegi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con, cls.path = F.korpus(DIM, kitaplar={"buyuk-kitap": ("textbook", SATIR)})
        cls.qv = (np.ones(DIM) / np.sqrt(DIM)).astype(np.float16)
        cls._encode, cls._device = embed.encode, embed.get_device
        embed.encode = lambda texts, device, **k: np.stack([cls.qv] * len(texts))
        embed.get_device = lambda pref="auto": "cpu"

    @classmethod
    def tearDownClass(cls):
        S.unload()
        embed.encode, embed.get_device = cls._encode, cls._device
        cls.con.close()
        os.unlink(cls.path)

    def test_puanlama_tum_matrisi_fp32ye_kopyalamaz(self):
        S.unload()
        S._fresh_matrix(self.con)                       # matrix loading cost is outside the measurement
        tum_fp32 = SATIR * DIM * 4                      # 5.1 MB: a one-shot astype allocates this
        tracemalloc.start()
        try:
            once, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            ids = S.dense_search(self.con, "q", 5)
            _, zirve = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(ids, F.kaba_kuvvet(self.con, self.qv, 5, "1=1", ()))
        self.assertLess(zirve - once, tum_fp32 // 4)


if __name__ == "__main__":
    unittest.main()
