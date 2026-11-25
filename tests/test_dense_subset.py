"""Filtered dense search must work without loading the whole vector matrix (2.5 GB in the corpus).

No model is loaded: the query vector is fixed (embed.encode is patched); what is tested is
which rows are loaded into memory and the returned ranking.

    cd <repo> && .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import unittest

import numpy as np

from tbmcp import embed, search as S
from tests import fixture as F

DIM = 16


class DenseSubset(unittest.TestCase):
    def setUp(self):
        self.con, self.path = F.korpus(DIM)
        self.qv = (np.ones(DIM) / np.sqrt(DIM)).astype(np.float16)
        self._encode, self._device, self._max = embed.encode, embed.get_device, getattr(S, "SUBSET_MAX_ROWS", None)
        embed.encode = lambda texts, device, **k: np.stack([self.qv] * len(texts))
        embed.get_device = lambda pref="auto": "cpu"
        S.SUBSET_MAX_ROWS = 50          # textbook collection (100 rows) is "large", notes (20) "small"
        S.unload()

    def tearDown(self):
        S.unload()
        embed.encode, embed.get_device = self._encode, self._device
        if self._max is not None:
            S.SUBSET_MAX_ROWS = self._max
        self.con.close()
        os.unlink(self.path)

    def test_kitap_aramasi_tum_matrisi_yuklemez(self):
        ids = S.dense_search(self.con, "q", 5, book="kitap-a")
        self.assertIsNone(S._matrix)
        self.assertEqual(ids, F.kaba_kuvvet(self.con, self.qv, 5, "b.slug=?", ("kitap-a",)))

    def test_kitap_aramasi_tum_matris_yoluyla_ayni_sonucu_verir(self):
        S._fresh_matrix(self.con)                       # as if an unfiltered search ran: full matrix in memory
        tum = S.dense_search(self.con, "q", 10, book="kitap-a")
        S.unload()
        alt = S.dense_search(self.con, "q", 10, book="kitap-a")
        self.assertIsNone(S._matrix)
        self.assertEqual(alt, tum)

    def test_esigi_asan_kitap_tum_matrisi_kullanir(self):
        ids = S.dense_search(self.con, "q", 5, book="kitap-b")        # 60 rows > threshold 50
        self.assertIsNotNone(S._matrix)
        self.assertEqual(ids, F.kaba_kuvvet(self.con, self.qv, 5, "b.slug=?", ("kitap-b",)))

    def test_kucuk_koleksiyon_tum_matrisi_yuklemez(self):
        ids = S.dense_search(self.con, "q", 5, collection="notes")
        self.assertIsNone(S._matrix)
        self.assertEqual(ids, F.kaba_kuvvet(self.con, self.qv, 5, "b.collection=?", ("notes",)))

    def test_buyuk_koleksiyon_tum_matrisi_kullanir(self):
        ids = S.dense_search(self.con, "q", 5, collection="textbook")
        self.assertIsNotNone(S._matrix)
        self.assertEqual(ids, F.kaba_kuvvet(self.con, self.qv, 5, "b.collection=?", ("textbook",)))

    def test_filtresiz_arama_tum_matrisi_kullanir(self):
        ids = S.dense_search(self.con, "q", 5)
        self.assertIsNotNone(S._matrix)
        self.assertEqual(ids, F.kaba_kuvvet(self.con, self.qv, 5, "1=1", ()))

    def test_kitap_ve_koleksiyon_birlikte_suzulur(self):
        self.assertEqual(S.dense_search(self.con, "q", 5, book="not-c", collection="textbook"), [])
        ids = S.dense_search(self.con, "q", 5, book="kitap-a", collection="textbook")
        self.assertEqual(ids, F.kaba_kuvvet(self.con, self.qv, 5, "b.slug=?", ("kitap-a",)))

    def test_bilinmeyen_kitap_bos_doner(self):
        self.assertEqual(S.dense_search(self.con, "q", 5, book="yok-boyle-kitap"), [])
        self.assertIsNone(S._matrix)

    def test_alt_kume_yeni_eklenen_vektoru_gorur(self):
        S.dense_search(self.con, "q", 5, book="kitap-a")             # the subset is now cached
        book_id = self.con.execute("SELECT id FROM books WHERE slug='kitap-a'").fetchone()[0]
        yeni = F.ekle(self.con, book_id, 99, "kitap-a yeni parca", self.qv.astype(np.float64))
        self.con.commit()
        self.assertEqual(S.dense_search(self.con, "q", 5, book="kitap-a")[0], yeni)

    def test_unload_alt_kumeyi_de_birakir(self):
        S.dense_search(self.con, "q", 5, book="kitap-a")
        self.assertTrue(S.loaded()["matris"])
        self.assertTrue(S.unload())
        self.assertFalse(S.loaded()["matris"])
        self.assertFalse(S.unload())


if __name__ == "__main__":
    unittest.main()
