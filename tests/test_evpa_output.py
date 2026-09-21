#!/usr/bin/env python3
"""Check the actual N/W outputs, derived products, readers and physical EVPA sticks."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np

from test_hdf5_contract import image_args

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kpolaris_image import load_image, evpa_tick_vectors


def run(command):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True,
                            env={**os.environ, "OMP_NUM_THREADS": "1"})
    if result.returncode:
        raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")


def linear_dataset(name):
    leaf = name.rsplit("/", 1)[-1]
    return (leaf in ("Q", "U", "Q_inv", "U_inv")
            or leaf.startswith(("dQ_", "dU_", "radial_stokes_Q_", "radial_stokes_U_",
                                "sum_Q_", "sum_U_", "flux_Q_", "flux_U_")))


def check_images(n_path, w_path, same=False):
    checked = []
    with h5py.File(n_path) as n, h5py.File(w_path) as w:
        assert n.attrs["evpa_0"] == "N"
        assert w.attrs["evpa_0"] == ("N" if same else "W")
        def check(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            sign = -1 if not same and linear_dataset(name) else 1
            np.testing.assert_array_equal(obj[...], sign * w[name][...], err_msg=name)
            checked.append(name)
        n.visititems(check)
        for name in ("sum_Q_inv", "sum_U_inv", "flux_Q_Jy", "flux_U_Jy"):
            if name in n["results"].attrs:
                assert n["results"].attrs[name] == (1 if same else -1) * w["results"].attrs[name]
        for fi in range(int(n.attrs["nfreq"])):
            group = n[f"frame_0/freq_{fi}"]
            assert group.attrs["evpa_0"] == "N"
            analysis = n.get("frame_0/analysis") if int(n.attrs["nfreq"]) == 1 else group.get("analysis")
            if analysis is not None and "physical_response" in analysis:
                for s in "IQUV":
                    parts = analysis[f"physical_response/source/{s}_inv"][...].sum(axis=0)
                    observed = group[f"{s}_inv"][...]
                    scale = max(np.abs(observed).max(), np.abs(parts).max(), 1e-300)
                    assert np.abs(parts - observed).max() / scale < 1e-10
    assert checked
    # Same polarization line in the image, independently of the reported angle.
    n, w = load_image(n_path), load_image(w_path)
    nv = evpa_tick_vectors(n, .5 * np.arctan2(n["stokes"][2], n["stokes"][1]))
    wv = evpa_tick_vectors(w, .5 * np.arctan2(w["stokes"][2], w["stokes"][1]))
    valid = n["valid"] & (np.hypot(n["stokes"][1], n["stokes"][2]) > 0)
    np.testing.assert_allclose(np.abs(nv[0]*wv[0] + nv[1]*wv[1])[valid], 1., atol=1e-14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-exe", type=Path, required=True)
    parser.add_argument("--trace-exe", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--analysis-enabled", action="store_true")
    parser.add_argument("--csv-enabled", action="store_true")
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    base = [x for x in image_args(work / "unused.h5")
            if not x.startswith(("--output=", "--evpa_0=", "--parameter_output="))]
    base += ["--parameter_output=auto"]
    for label, setting in (("default", []), ("N", ["--evpa_0=N"]), ("W", ["--evpa_0=W"])):
        run([args.image_exe, *base, *setting, f"--output={work / (label + '.h5')}"])
        assert f"evpa_0={'W' if label == 'W' else 'N'}\n" in (work / (label + '.h5.params')).read_text()
    check_images(work / "default.h5", work / "N.h5", same=True)
    check_images(work / "N.h5", work / "W.h5")
    rejected = subprocess.run([str(args.image_exe), "--evpa_0=bad", f"--output={work/'bad.h5'}"], capture_output=True)
    assert rejected.returncode and b"evpa_0 must be N or W" in rejected.stderr
    assert not (work / "bad.h5").exists()
    if args.analysis_enabled:
        for zero in ("N", "W"):
            run([args.image_exe, *base, f"--evpa_0={zero}", "--analysis_mode=1",
                 "--analysis_radial_bins=4", "--analysis_partition=near_far",
                 "--analysis_response=temperature_scale", "--freq_list=230e9,345e9",
                 f"--output={work / ('analysis_' + zero + '.h5')}"])
        check_images(work / "analysis_N.h5", work / "analysis_W.h5")
    # Both file conventions must convert to the same chosen comparison basis.
    h5dump = shutil.which("h5dump")
    if h5dump and (ROOT/"scripts/arcmancer_h5_to_stokes_csv.py").exists():
        for target in ("camera", "ipole-native"):
            contents = []
            for zero in ("N", "W"):
                dest = work / f"{zero}_{target}.csv"
                run([sys.executable, ROOT/"scripts/arcmancer_h5_to_stokes_csv.py",
                     work/f"{zero}.h5", dest, "--qu-conv", target, "--h5dump", h5dump])
                with dest.open() as handle:
                    contents.append(list(csv.DictReader(line for line in handle if not line.startswith("#"))))
            assert contents[0] == contents[1]
            # Check actual values, not only agreement between two conversions.
            with h5py.File(work/("W.h5" if target == "camera" else "N.h5")) as h:
                for row in contents[0]:
                    for component in "IQUV":
                        expected = h[f"frame_0/freq_0/{component}"][int(row["iy"]), int(row["ix"])]
                        np.testing.assert_allclose(float(row[component+"_nu"]), expected, rtol=1e-14, atol=0)
        if args.analysis_enabled:
            dest = work / "second_frequency.csv"
            run([sys.executable, ROOT/"scripts/arcmancer_h5_to_stokes_csv.py",
                 work/"analysis_N.h5", dest, "--qu-conv=ipole-native", "--freq=1", "--h5dump", h5dump])
            assert "# frequency_hz,345000000000\n" in dest.read_text()
        conflict = work / "conflicting_basis.h5"
        shutil.copyfile(work/"N.h5", conflict)
        with h5py.File(conflict, "r+") as h:
            h["frame_0/freq_0"].attrs["evpa_0"] = "W"
        result = subprocess.run([sys.executable, str(ROOT/"scripts/arcmancer_h5_to_stokes_csv.py"),
                                 str(conflict), str(work/"rejected.csv"), "--h5dump", h5dump], capture_output=True)
        assert result.returncode and b"conflicting evpa_0" in result.stderr
    if args.csv_enabled:
        for zero in ("N", "W"):
            dest = work / f"direct_{zero}.csv"
            run([args.image_exe, *base, f"--evpa_0={zero}", "--format=csv", f"--output={dest}"])
            with dest.open() as handle, h5py.File(work/f"{zero}.h5") as h:
                for row in csv.DictReader(line for line in handle if not line.startswith("#")):
                    for s in "IQUV":
                        np.testing.assert_allclose(float(row[s+"_nu"]), h[f"frame_0/freq_0/{s}"][int(row["iy"]),int(row["ix"])], rtol=1e-14)
    if args.trace_exe:
        for zero in ("N", "W"):
            run([args.trace_exe, *base, f"--evpa_0={zero}", "--trace_mode=single", "--ix=4", "--iy=4",
                 "--trace_fields=all", "--trace_precision=double", "--max_trace_samples=8192",
                 f"--output={work / ('trace_' + zero + '.h5')}"])
        with h5py.File(work/"trace_N.h5") as n, h5py.File(work/"trace_W.h5") as w, h5py.File(work/"N.h5") as im:
            for s in "IQUV":
                nv, wv = n[f"rays/final_observed_{s}_inv"][...], w[f"rays/final_observed_{s}_inv"][...]
                np.testing.assert_array_equal(nv, (-1 if s in "QU" else 1)*wv)
                np.testing.assert_allclose(nv[0], im[f"frame_0/freq_0/{s}_inv"][4,4], rtol=1e-10, atol=1e-300)
                np.testing.assert_array_equal(n[f"trace/S{s}"][...], w[f"trace/S{s}"][...])
            q, u = (n[f"rays/final_observed_{s}_inv"][...] for s in "QU")
            np.testing.assert_array_equal(n["derived/final_evpa_wrapped_rad"][...], .5*np.arctan2(u,q))
            np.testing.assert_array_equal(n["derived/evpa_unwrapped_rad"][...], w["derived/evpa_unwrapped_rad"][...])
    print("EVPA N/W output and available diagnostic/reader checks passed")


if __name__ == "__main__":
    main()
