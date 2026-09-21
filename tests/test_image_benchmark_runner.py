#!/usr/bin/env python3
"""Guard CPU placement and executable RSS against misleading benchmarks."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_image_scaling.py"
spec = importlib.util.spec_from_file_location("image_benchmark", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class BenchmarkAccounting(unittest.TestCase):
    def test_physical_cores_precede_siblings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for cpu in range(16):
                topology = root / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text("0")
                (topology / "core_id").write_text(str(cpu // 2))
            self.assertEqual(runner.physical_core_first(range(16), root),
                             list(range(0,16,2)) + list(range(1,16,2)))
            self.assertEqual(runner.physical_core_first([1,2,3,5,7], root), [1,2,5,7,3])
            self.assertEqual(runner.physical_core_first([], root), [])

    def test_rejects_multiple_frequency_products(self):
        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "multifrequency.h5"
            with runner.h5py.File(image,"w") as h:
                for index in range(2):
                    group = h.create_group(f"frame_0/freq_{index}")
                    for component,value in zip("IQUV",[1.,.1,.05,.01]):
                        group.create_dataset(component,data=runner.np.full((4,4),value))
                h.create_dataset("frame_0/diagnostics/reason",data=runner.np.ones((4,4)))
            with self.assertRaisesRegex(ValueError,"one frequency"):
                runner.check_image(image,4,False)

    def test_rss_excludes_launcher(self):
        # A fresh helper makes the inherited parent RSS explicit. The launched
        # sleep executable needs only a few MiB, irrespective of this buffer.
        code = r"""
import importlib.util, os, pathlib, sys
spec=importlib.util.spec_from_file_location('runner',sys.argv[1])
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
buffer=bytearray(128*2**20)
r=m.measure(['/bin/sleep','.25'],os.environ.copy(),pathlib.Path(sys.argv[2]),False,[])
assert r['returncode']==0
assert 0 < r['host_peak_rss_bytes'] < 32*2**20, r
assert r['host_wait4_maxrss_bytes'] > r['host_peak_rss_bytes'] + 64*2**20, r
assert r['host_sampled_peak_swap_bytes']==0
"""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable,"-c",code,str(SCRIPT),str(Path(d)/"sleep.h5")],check=True)

if __name__ == "__main__":
    unittest.main()
