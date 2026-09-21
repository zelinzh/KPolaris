#!/usr/bin/env python3
"""Check native image interpretation and the public workflow's failure contract."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kpolaris_image import (image_extent, image_summary, load_image, rotate_stokes_basis,
                            ipole_to_camera_stokes, evpa_tick_vectors)
from plot_kpolaris_pol import polarization_maps
from plot_trace_diagnostics import trace_dataset, derived_group


class UserWorkflows(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kpolaris workflow ")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.image = self.directory / "native image.h5"
        with h5py.File(self.image, "w") as h:
            h.attrs.update(image_width_x_M=20., image_width_y_M=10.,
                           camera="parallel", x_offset=2., y_offset=-1.,
                           frequency_hz=230e9, intensity_to_flux_jy_per_pixel=2.,
                           flux_scale_valid=1)
            for frame in range(2):
                f = h.create_group(f"frame_{frame}")
                f.create_group("diagnostics")["reason"] = np.ones((2, 3), dtype=int)
                for frequency in range(2):
                    g = f.create_group(f"freq_{frequency}")
                    g.attrs.update(frequency_hz=(230 + 115 * frequency) * 1e9,
                                   intensity_to_flux_jy_per_pixel=2. + frequency)
                    for i, s in enumerate("IQUV"):
                        g[s] = np.full((2, 3), (1 + 2 * frame + frequency) * [1., .3, .4, -.1][i])
                    g.create_group("analysis")["absorption_depth"] = np.ones((2, 3)) * (1 + frequency)

    def command(self, script, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / script),
                               *map(str, arguments)], capture_output=True, text=True,
                              env={**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"})

    def test_selected_frame_frequency_and_net_stokes(self):
        result = image_summary(load_image(self.image, 1, 1))
        self.assertTrue(result["ok"])
        self.assertEqual(result["frequency_hz"], 345e9)
        self.assertEqual(result["flux_valid_pixels_jy"]["I"], 72.)
        self.assertAlmostEqual(result["net_linear_fraction"], .5)
        self.assertAlmostEqual(result["net_circular_fraction"], -.1)
        self.assertAlmostEqual(result["camera_evpa_deg"], 26.5650511771)
        self.assertEqual(image_extent(load_image(self.image))[0], [-8., 12., -6., 4.])

    def test_no_fabricated_flux_or_angular_scale(self):
        with h5py.File(self.image, "r+") as h:
            h.attrs["flux_scale_valid"] = 0
        image = load_image(self.image)
        self.assertIsNone(image_summary(image)["flux_valid_pixels_jy"])
        with self.assertRaisesRegex(ValueError, "angular scale"):
            image_extent(image, "muas")
        extent, unit = image_extent(image, "muas", distance_pc=8100., mass_solar=4.3e6)
        self.assertGreater(extent[1] - extent[0], 100.)
        self.assertIn("as", unit)
        image["attrs"]["mbh_solar"] = 4.3e6
        image["attrs"]["dsource"] = 8100.
        self.assertEqual(image_extent(image, "muas")[0], extent)

    def test_bad_rays_and_missing_termination_are_reported(self):
        with h5py.File(self.image, "r+") as h:
            h["frame_0/diagnostics/reason"][0, 0] = 2
            h["frame_0/freq_0/Q"][1, 1] = np.nan
        result = image_summary(load_image(self.image))
        self.assertFalse(result["ok"])
        self.assertEqual(result["valid_pixels"], 4)
        self.assertEqual(result["nonfinite_pixels"], 1)
        p = self.command("kpolaris.py", "inspect", self.image, "--json")
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertFalse(json.loads(p.stdout)["ok"])
        with h5py.File(self.image, "r+") as h:
            del h["frame_1/diagnostics"]
        result = image_summary(load_image(self.image, 1))
        self.assertFalse(result["termination_available"])
        self.assertFalse(result["ok"])

    def test_missing_groups_and_malformed_shapes_fail(self):
        with self.assertRaisesRegex(ValueError, "available images"):
            load_image(self.image, 3)
        with self.assertRaises(ValueError):
            load_image(self.image, -1)
        with h5py.File(self.image, "r+") as h:
            del h["frame_0/freq_0/U"]
            h["frame_0/freq_0/U"] = np.ones((3, 2))
        with self.assertRaisesRegex(ValueError, "same shape"):
            load_image(self.image)

    def test_weak_or_unpolarized_pixels_have_no_displayed_angle(self):
        stokes = np.array([[[1., 1e-6, 0., 1.]], [[.3, 1e-6, 0., 0.]],
                           [[.4, 0., 0., 0.]], [[-.1, 0., 0., 0.]]])
        lp, angle, cp = polarization_maps(stokes, np.ones((1, 4), bool), .001)
        self.assertEqual(lp[0, 0], 50.)
        self.assertEqual(cp[0, 0], -10.)
        self.assertTrue(np.all(np.isnan(angle[0, 1:])))
        self.assertTrue(np.isnan(lp[0, 1]))

    def test_basis_rotation_from_electric_fields_and_ipole_metadata(self):
        field = np.array([.7 + .2j, -.3 + .4j])
        def stokes(e):
            cross = e[0] * e[1].conjugate()
            return np.array([np.abs(e[0])**2 + np.abs(e[1])**2,
                             np.abs(e[0])**2 - np.abs(e[1])**2, 2*cross.real, -2*cross.imag])
        native = stokes(field)
        for angle in (0., 27., 90., -43.):
            c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            expected = stokes(np.array([[c,s],[-s,c]]) @ field)
            np.testing.assert_allclose(rotate_stokes_basis(native, angle), expected, atol=1e-14)
        n = native.copy(); n[1:3] *= -1
        np.testing.assert_array_equal(ipole_to_camera_stokes(n, 'N'), native)
        np.testing.assert_array_equal(ipole_to_camera_stokes(native, b'W'), native)
        for unknown in ('', 'camera', 'unknown'):
            with self.assertRaises(ValueError): ipole_to_camera_stokes(native, unknown)

    def test_pinhole_evpa_segments_follow_gnomonic_projection(self):
        image = load_image(self.image)
        image['attrs'].update(camera='pinhole', fov=1.2, fovy=.8, x_offset=.13, y_offset=-.07)
        angle = np.full((2,3), .7)
        vx, vy = evpa_tick_vectors(image, angle)
        # Finite-difference the image location of a ray displaced in its
        # electric-field direction: independent of the analytic Jacobian.
        for row in range(2):
            for col in range(3):
                sx=(col+.5)/3*1.2-.6+.13; sy=(row+.5)/2*.8-.4-.07
                n=np.array([sx,sy,1.]); n/=np.linalg.norm(n)
                e1=np.array([1.,0,0])-n*n[0]; e1/=np.linalg.norm(e1)
                e2=np.cross(n,e1)
                e=np.cos(.7)*e1+np.sin(.7)*e2
                plus=n+1e-6*e; minus=n-1e-6*e
                d=plus[:2]/plus[2]-minus[:2]/minus[2]; d/=np.linalg.norm(d)
                np.testing.assert_allclose([vx[row,col],vy[row,col]],d,atol=1e-10)

    def test_unsupported_sky_labels_and_file_basis_do_not_change_plot(self):
        for option in ('--evpa-conv=EofN', '--evpa-conv=NofW', '--qu-conv=ipole-native'):
            output = self.directory / 'unjustified.pdf'
            result = self.command('plot_kpolaris_pol.py', self.image, option, f'--output={output}')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
        with h5py.File(self.image, 'r+') as h:
            h.attrs['evpa_0'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'unknown evpa_0'):
            load_image(self.image)

    def test_n_w_output_conventions_preserve_physical_ticks(self):
        image = load_image(self.image)
        image['attrs']['evpa_0'] = 'W'
        angle = np.full((2,3), .7)
        wx, wy = evpa_tick_vectors(image, angle)
        image['attrs']['evpa_0'] = 'N'
        nx, ny = evpa_tick_vectors(image, angle - np.pi/2)
        np.testing.assert_allclose(nx, wx, atol=1e-14)
        np.testing.assert_allclose(ny, wy, atol=1e-14)
        with h5py.File(self.image, 'r+') as h:
            h.attrs['evpa_0'] = 'N'
        loaded = load_image(self.image)
        self.assertEqual(loaded['attrs']['evpa_0'], 'N')
        np.testing.assert_array_equal(loaded['stokes'], image['stokes'])

    def test_evpa_segments_respect_rectangular_pixel_scale(self):
        image = load_image(self.image)
        angle = np.full((2,3), np.pi/4)
        vx, vy = evpa_tick_vectors(image, angle, 'pixel')
        # Fixture pixels are 20/3 M wide, 10/2 M high.
        np.testing.assert_allclose(vy/vx, (2/10)/(3/20))

    def test_plot_selection_scale_summary_and_parent_directories(self):
        out = self.directory / "figures/selected.pdf"
        summary = self.directory / "records/selected.json"
        p = self.command("plot_kpolaris_pol.py", self.image, "--frame=1", "--freq-index=1",
                         "--layout=stokes", "--scale=5", f"--output={out}", f"--summary-json={summary}")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(out.read_bytes().startswith(b"%PDF"))
        self.assertEqual(json.loads(summary.read_text())["flux_valid_pixels_jy"]["I"], 120.)
        with h5py.File(self.image, "r") as h:
            self.assertEqual(h["frame_1/freq_1"].attrs["intensity_to_flux_jy_per_pixel"], 3.)

    def test_analysis_frequency_selection_ignores_legacy_alias(self):
        with h5py.File(self.image, "r+") as h:
            h.create_group("frame_1/analysis")["wrong_field"] = np.ones((2, 3))
        out = self.directory / "maps/selected.png"
        p = self.command("plot_analysis_maps.py", self.image, "--frame=1", "--freq-index=1",
                         "--fields=absorption_depth", f"--output={out}")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(out.read_bytes().startswith(b"\x89PNG"))

    def test_intensity_brightness_display_preserves_physical_flux_and_input(self):
        before = self.image.read_bytes()
        out = self.directory / "intensity.png"
        summary = self.directory / "intensity.json"
        p = self.command("plot_kpolaris_pol.py", self.image, "--frame=1", "--freq-index=1",
                         "--layout=intensity", "--intensity-unit=brightness-temperature",
                         "--evpa-ticks", f"--output={out}", f"--summary-json={summary}")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(out.read_bytes().startswith(b"\x89PNG"))
        result = json.loads(summary.read_text())
        self.assertEqual(result["flux_valid_pixels_jy"]["I"], 72.)
        self.assertEqual(result["display_intensity_unit"], "brightness-temperature")
        self.assertEqual(self.image.read_bytes(), before)
        p = self.command("plot_kpolaris_pol.py", self.image,
                         "--intensity-unit=brightness-temperature", "--scale=5")
        self.assertNotEqual(p.returncode, 0)

    def test_trace_frequency_selection_does_not_use_frequency_zero_alias(self):
        with h5py.File(self.image, "r+") as h:
            h.create_group("derived")["depth"] = [1.]
            h.create_group("trace/freq_1/derived")["depth"] = [2.]
            h.create_group("trace/shared")["lambda"] = [3.]
            h["trace/nu"] = [230e9]
            h["trace/freq_1/nu"] = [345e9]
            self.assertEqual(trace_dataset(h, "nu", 1)[0], 345e9)
            self.assertEqual(trace_dataset(h, "derived/depth", 1)[0], 2.)
            self.assertEqual(trace_dataset(h, "lambda", 1)[0], 3.)
            self.assertIsNone(derived_group(h, 2))
            self.assertIsNone(trace_dataset(h, "nu", 2))

    def test_dry_run_is_literal_and_creates_no_files(self):
        out = self.directory / "new output $literal"
        build = self.directory / "new build"
        p = self.command("kpolaris.py", "quickstart", "--json", "--dry-run",
                         f"--output-dir={out}", f"--build-dir={build}")
        self.assertEqual(p.returncode, 0, p.stderr)
        result = json.loads(p.stdout)
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["ok"])
        commands = result["commands"]
        self.assertTrue(any(f"--output={out / 'image.h5'}" in c["command"] for c in commands))
        self.assertTrue(any("--nx=256" in c["command"] for c in commands))
        self.assertTrue(any("--layout=intensity" in c["command"] for c in commands))
        self.assertFalse(out.exists())
        self.assertFalse(build.exists())
        p = self.command("kpolaris.py", "quickstart", "--backend=cuda", "--cuda-arch=AMPERE80",
                         "--dry-run", "--json", f"--build-dir={build}")
        self.assertEqual(p.returncode, 0, p.stderr)
        plan = json.loads(p.stdout)["commands"]
        self.assertTrue(any("-DKPOLARIS_CUDA_ARCH=AMPERE80" in c["command"] for c in plan))

    def test_failure_and_nonempty_output_are_never_success(self):
        out = self.directory / "failed-run"
        p = self.command("kpolaris.py", "demo", "--image-exe=/does/not/exist",
                         f"--output-dir={out}", "--no-plot", "--json")
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertFalse(json.loads(p.stdout)["ok"])
        self.assertFalse(json.loads((out / "run.json").read_text())["ok"])
        previous = (out / "run.json").read_bytes()
        p = self.command("kpolaris.py", "demo", f"--output-dir={out}", "--json")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("not empty", json.loads(p.stdout)["error"])
        self.assertEqual((out / "run.json").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
