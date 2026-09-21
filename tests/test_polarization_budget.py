#!/usr/bin/env python3
"""Physical counterexamples and observational-response tests for source tags."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_polarization_budget import cancellation_budget, radial_amplitude_maps, analyze_image


class PolarizationBudgetTests(unittest.TestCase):
    def tags(self):
        # Two radial bins, two pixels. I tags are all physical; Q/U cancel
        # partly within the first ray and again between the two image pixels.
        return np.array([[[2., 2.], [2., 2.]],
                         [[1., -1.], [-.5, .2]],
                         [[.2, .1], [.1, -.2]],
                         [[.1, -.2], [-.1, .1]]])

    def test_exact_hierarchical_budget_and_coarsening(self):
        tags = self.tags()
        b = cancellation_budget(tags.sum(axis=1), tags)
        for name in ("linear", "circular"):
            c = b[name]
            self.assertAlmostEqual(c["los_cancellation_fraction"] + c["image_cancellation_fraction"] + c["surviving_fraction"], 1)
            self.assertAlmostEqual(c["tagged_fraction"] * c["los_coherence"] * c["image_coherence"], c["net_fraction"])
        self.assertAlmostEqual(b["radial_coarsening"][-1]["linear_los_coherence"], 1)
        self.assertLess(b["linear"]["los_coherence"], 1)

    def test_complete_cancellation_is_retained(self):
        tags = self.tags()
        tags[1:, 1, 0] = -tags[1:, 0, 0]
        obs = tags.sum(axis=1)
        b = cancellation_budget(obs, tags)
        expected = np.hypot(obs[1], obs[2]).sum() / np.hypot(tags[1], tags[2]).sum()
        self.assertAlmostEqual(b["linear"]["los_coherence"], expected)
        with tempfile.TemporaryDirectory() as d:
            with h5py.File(Path(d) / 'tags.h5', 'w') as h5:
                for i, name in enumerate("QUV", 1):
                    h5.create_dataset(f"radial_stokes_{name}_inv_contribution", data=tags[i, :, None, :])
                linear, circular = radial_amplitude_maps(h5)
                self.assertGreater(linear[0, 0], 0)
                self.assertGreater(circular[0, 0], 0)

    def test_rotation_and_extreme_unit_scaling(self):
        tags = self.tags()
        reference = cancellation_budget(tags.sum(axis=1), tags)
        p = (tags[1] + 1j * tags[2]) * np.exp(2j * .73)
        tags[1], tags[2] = p.real, p.imag
        for scale in (1e-200, 1e200):
            b = cancellation_budget(tags.sum(axis=1) * scale, tags * scale)
            for key in ("los_coherence", "image_coherence", "net_fraction"):
                self.assertAlmostEqual(b["linear"][key], reference["linear"][key])

    def test_source_response_matches_independent_finite_difference(self):
        tags = self.tags()
        b = cancellation_budget(tags.sum(axis=1), tags)
        eps = 1e-5
        for k, response in enumerate(b["source_response"]):
            def observables(sign):
                perturbed = tags.copy()
                perturbed[:, k] *= 1 + sign * eps
                i, q, u, v = perturbed.sum(axis=(1, 2))
                return np.array([np.log(np.hypot(q, u)), .5*np.arctan2(u, q), np.hypot(q,u)/i, v/i])
            fd = (observables(1) - observables(-1)) / (2 * eps)
            exact = [response[key] for key in (
                "dln_net_linear_amplitude_d_emission_scale", "d_net_evpa_rad_d_emission_scale",
                "d_net_linear_fraction_d_emission_scale", "d_net_circular_fraction_d_emission_scale")]
            np.testing.assert_allclose(fd, exact, rtol=1e-8, atol=1e-9)

    def test_undefined_polarization_and_bad_closure(self):
        tags = self.tags(); tags[1:] = 0
        b = cancellation_budget(tags.sum(axis=1), tags)
        self.assertIsNone(b["linear"]["los_coherence"])
        self.assertFalse(b["source_response_linear_valid"])
        json.dumps(b, allow_nan=False)
        obs = tags.sum(axis=1); obs[0,0] += 1
        with self.assertRaisesRegex(ValueError, 'closure'):
            cancellation_budget(obs, tags)

    def test_frequency_selection_valid_mask_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'image.h5'
            tags = self.tags()
            with h5py.File(path, 'w') as h5:
                h5.attrs['nfreq'] = 2
                frame = h5.create_group('frame_0')
                frame.create_dataset('diagnostics/reason', data=[[1, 2]])
                for fi in (0, 1):
                    freq = frame.create_group(f'freq_{fi}')
                    freq.attrs['frequency_hz'] = (86e9, 230e9)[fi]
                    for i, s in enumerate('IQUV'):
                        freq.create_dataset(s+'_inv', data=tags[i].sum(axis=0)[None,:])
                    a = freq.create_group('analysis'); a.attrs['observer_weighted_available'] = 1
                    a.create_dataset('radial_bin_edges', data=[2., 4., 8.])
                    for i, s in enumerate('IQUV'):
                        a.create_dataset(f'radial_stokes_{s}_inv_contribution', data=tags[i,:,None,:])
            b = analyze_image(path, 1)
            self.assertEqual(b['frequency_hz'], 230e9)
            self.assertEqual(b['valid_pixels'], 1)
            self.assertEqual(b['excluded_pixels'], 1)
            with h5py.File(path, 'a') as h5:
                del h5['frame_0/freq_1/analysis']
                h5.create_group('frame_0/analysis')
            with self.assertRaisesRegex(ValueError, 'independent analysis'):
                analyze_image(path, 1)


if __name__ == '__main__':
    unittest.main()
