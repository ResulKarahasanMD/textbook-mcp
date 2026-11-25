"""MPS memory: the models must stay at fp16 size, and the cache must be released after a search.

Loads the real models (bge-m3 + bge-reranker-v2-m3, ~20 s), so it is off by default:

    cd <repo> && TEXTBOOK_GPU_TESTS=1 .venv/bin/python -m unittest tests.test_gpu_memory -v
"""
from __future__ import annotations

import os
import unittest

from tests import fixture as F

MB = 1048576


@unittest.skipUnless(os.environ.get("TEXTBOOK_GPU_TESTS") == "1", "TEXTBOOK_GPU_TESTS=1 ile acilir")
class GpuBellek(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("MPS yok")
        from tbmcp import search as S
        cls.torch, cls.S = torch, S
        cls.con, cls.path = F.korpus(dim=1024)
        S.unload()
        cls.hits = S.search(cls.con, "insulin glucose", top_k=2, dense=True, use_rerank=True)
        cls.yuklu = S.loaded()
        cls.current = torch.mps.current_allocated_memory()
        cls.driver = torch.mps.driver_allocated_memory()

    @classmethod
    def tearDownClass(cls):
        cls.S.unload()
        cls.con.close()
        os.unlink(cls.path)

    def test_arama_iki_modeli_de_kullandi(self):
        self.assertEqual(len(self.hits), 2)
        self.assertIn("ce", self.hits[0].why)
        self.assertTrue(self.yuklu["model"] and self.yuklu["reranker"])

    def test_iki_model_fp16_boyutunda_durur(self):
        # 2 x 568 M parameters x 2 bytes = 2166 MiB. Loading fp32 and calling .half() on the
        # device left the tensors in fp32-sized buffers (measured: 3369 MiB).
        self.assertLess(self.current / MB, 2500)

    def test_arama_sonrasi_geri_verilebilir_onbellek_kalmaz(self):
        self.torch.mps.empty_cache()
        geri_verilen = self.driver - self.torch.mps.driver_allocated_memory()
        self.assertLess(geri_verilen / MB, 64)


if __name__ == "__main__":
    unittest.main()
