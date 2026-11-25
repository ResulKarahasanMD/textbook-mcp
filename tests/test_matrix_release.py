"""An unloaded vector matrix must be returned to the operating system.

macOS libmalloc keeps freed large blocks in its "large cache", and they stay in the
process footprint (measured: a deleted 2.3 GB np.empty remained at 2734 MB in `footprint`;
the same block allocated with anonymous mmap dropped once deleted). So whether idle
unloading (unload) means anything depends on how memory is allocated. The test measures
the real footprint with /usr/bin/footprint.
"""
from __future__ import annotations

import gc
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import unittest

import numpy as np

from tbmcp import embed, search as S

SATIR, DIM = 60_000, 1024                     # fp16 matrix 117 MiB
BOYUT_MB = SATIR * DIM * 2 / 1048576


def ayak_izi_mb() -> float:
    out = subprocess.run(["/usr/bin/footprint", str(os.getpid())], capture_output=True, text=True).stdout
    sayi, birim = re.search(r"phys_footprint: (\d+) (B|KB|MB|GB)", out).groups()
    return float(sayi) * {"B": 1 / 1048576, "KB": 1 / 1024, "MB": 1, "GB": 1024}[birim]


@unittest.skipUnless(os.path.exists("/usr/bin/footprint"), "macOS only")
class MatrisBirakma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        con = sqlite3.connect(cls.path)
        con.executescript(open(os.path.join(os.path.dirname(__file__), "..", "tbmcp", "schema.sql")).read())
        con.execute("INSERT INTO books(id,slug,title,path,collection) VALUES(1,'tek-kitap','t','/x.pdf','textbook')")
        rng = np.random.default_rng(3)
        for i in range(0, SATIR, 5000):
            v = rng.standard_normal((5000, DIM)).astype(np.float16)
            con.executemany("INSERT INTO chunks(id,book_id,page_start,page_end,text) VALUES(?,1,1,1,'x')",
                            [(i + j + 1,) for j in range(5000)])
            con.executemany("INSERT INTO embeddings(chunk_id,vec) VALUES(?,?)",
                            [(i + j + 1, v[j].tobytes()) for j in range(5000)])
        con.commit()
        con.close()
        cls.con = sqlite3.connect(f"file:{cls.path}?mode=ro", uri=True, check_same_thread=False)

    @classmethod
    def tearDownClass(cls):
        S.unload()
        cls.con.close()
        os.unlink(cls.path)

    def _bekle_ve_olc(self) -> float:
        gc.collect()
        time.sleep(1)
        return ayak_izi_mb()

    def test_tum_matris_silinince_bellek_doner(self):
        once = self._bekle_ve_olc()
        ids, mat = embed.load_matrix(self.con)
        dolu = ayak_izi_mb()
        del ids, mat
        sonra = self._bekle_ve_olc()
        # Blocks released by earlier allocations and kept by the allocator can be reused, so
        # the increase relative to `once` is not reliable; what is measured is how much deletion gives back.
        self.assertGreater(dolu - sonra, 0.8 * BOYUT_MB, f"before {once:.0f}, loaded {dolu:.0f}, after {sonra:.0f}")

    def test_alt_kume_unload_ile_bellek_doner(self):
        S.unload()
        once = self._bekle_ve_olc()
        self.assertIsNotNone(S._subset_matrix(self.con, "tek-kitap", None))
        dolu = ayak_izi_mb()
        S.unload()
        sonra = self._bekle_ve_olc()
        self.assertGreater(dolu - sonra, 0.8 * BOYUT_MB, f"before {once:.0f}, loaded {dolu:.0f}, after {sonra:.0f}")


if __name__ == "__main__":
    unittest.main()
