#!/usr/bin/env python3
"""Tests for the external Combi--Ressler trajectory generator."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_binary_trajectory.py"
)
SPEC = importlib.util.spec_from_file_location("generate_binary_trajectory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class BinaryTrajectoryGeneratorTests(unittest.TestCase):
    def test_merger_completion_cannot_be_forged_by_cadence_or_early_cut(self) -> None:
        time = np.asarray([0.0, 1.0, 2.0])
        position = np.zeros((3, 2, 3), dtype=np.float64)
        position[:, 0, 0] = [0.5, 0.25, 0.1000000005]
        position[:, 1, 0] = -position[:, 0, 0]
        velocity = np.zeros_like(position)
        velocity[:, 0, 1] = 0.45
        velocity[:, 1, 1] = -0.45
        raw = {
            "time": time,
            "position": position,
            "velocity": velocity,
            "separation": np.linalg.norm(position[:, 0] - position[:, 1], axis=1),
        }
        args = SimpleNamespace(
            merger_time_M=None,
            transition_velocity="consistent",
            print_step=10**9,
            dt_M=10**6,
            rmin_M=0.2,
            allow_truncated_inspiral=False,
        )
        cadence = generator.merger_reach_diagnostic(raw, args)
        self.assertFalse(cadence["reached"])
        self.assertAlmostEqual(cadence["selected_merger_separation"], 0.200000001)
        self.assertEqual(
            cadence["numerical_tolerance"],
            generator.MERGER_REACH_ABSOLUTE_TOLERANCE_M,
        )

        # The final source sample can be below rmin while an explicitly chosen
        # earlier merger state is still well outside it.  Completion follows
        # the state that is actually fed to the transition, not the file tail.
        position[-1, 0, 0] = 0.05
        position[-1, 1, 0] = -0.05
        raw["separation"] = np.linalg.norm(
            position[:, 0] - position[:, 1], axis=1
        )
        args.merger_time_M = 1.0
        early = generator.merger_reach_diagnostic(raw, args)
        self.assertFalse(early["reached"])
        self.assertAlmostEqual(early["selected_merger_separation"], 0.5)
        self.assertAlmostEqual(early["final_separation"], 0.1)
        with self.assertRaisesRegex(ValueError, "selected merger state"):
            generator.validate_merger_reached(raw, args)
        args.allow_truncated_inspiral = True
        self.assertFalse(generator.validate_merger_reached(raw, args)["reached"])

    def test_equal_nonspinning_remnant_tracks_rotated_orbital_axis(self) -> None:
        args = generator.make_parser().parse_args(["--dry-run"])
        raw = {
            "time": np.asarray([0.0, 1.0, 2.0]),
            "orbital_angular_momentum": np.asarray(
                [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
            ),
        }
        remnant = generator.resolve_remnant(args, raw)
        np.testing.assert_allclose(remnant["spin_chi"], [0.0, 0.68646, 0.0])
        self.assertEqual(
            remnant["orientation"]["policy"],
            "aligned-with-orbital-L-at-transition-start",
        )

    def test_paper_nr_fit_formula_regressions_are_self_contained(self) -> None:
        axis = np.asarray([0.0, 0.0, 1.0])
        zero = np.zeros(3)
        self.assertAlmostEqual(
            generator.paper_nr_final_mass(1.0, zero, zero, axis),
            0.95173,
            places=14,
        )
        self.assertAlmostEqual(
            generator.paper_nr_final_spin_magnitude(1.0, zero, zero, axis),
            0.68646,
            places=14,
        )
        np.testing.assert_array_equal(
            generator.paper_nr_kick_components(1.0, zero, zero, axis, 0.37),
            0.0,
        )

        # Fixed generic configuration, independently evaluated against the
        # Zenodo v1 precession.py formulas (BMR12 and HBR16_34corr).
        q = 0.5
        theta1 = np.arccos(0.3)
        theta2 = np.arccos(-0.4)
        delta_phi = 1.2
        spin1 = 0.7 * np.asarray(
            [np.sin(theta1), 0.0, np.cos(theta1)]
        )
        spin2 = 0.4 * np.asarray(
            [
                np.sin(theta2) * np.cos(delta_phi),
                np.sin(theta2) * np.sin(delta_phi),
                np.cos(theta2),
            ]
        )
        self.assertAlmostEqual(
            generator.paper_nr_final_mass(q, spin1, spin2, axis),
            0.9560013454998099,
            places=14,
        )
        self.assertAlmostEqual(
            generator.paper_nr_final_spin_magnitude(q, spin1, spin2, axis),
            0.7452111456013830,
            places=14,
        )
        kick = generator.paper_nr_kick_components(
            q, spin1, spin2, axis, 0.37
        )
        np.testing.assert_allclose(
            kick,
            [
                0.0003427201423758923,
                0.00012603797021953992,
                0.0046491739915940902,
            ],
            rtol=2.0e-12,
            atol=2.0e-15,
        )
        self.assertTrue(np.all(np.isfinite(kick)))
        self.assertLess(float(np.linalg.norm(kick)), 1.0)
        other_phase = generator.paper_nr_kick_components(
            q, spin1, spin2, axis, 2.1
        )
        self.assertNotEqual(float(kick[2]), float(other_phase[2]))
        np.testing.assert_array_equal(
            kick,
            generator.paper_nr_kick_components(q, spin1, spin2, axis, 0.37),
        )

    def test_paper_nr_fit_requires_an_explicit_kick_phase(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--dry-run",
                "--remnant-model=paper-nr-fit",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--remnant-kick-phase-rad", result.stderr)
        self.assertIn("no random phase", result.stderr)

    def test_equal_nonspinning_remnant_checks_actual_dat_spins(self) -> None:
        raw = {"spin_chi": np.zeros((2, 2, 3), dtype=np.float64)}
        raw["spin_chi"][1, 0, 1] = 0.2
        with self.assertRaisesRegex(ValueError, "actual DAT spin columns"):
            generator.validate_raw_remnant_compatibility(
                SimpleNamespace(remnant_model="equal-nonspinning"), raw
            )

    def test_external_dat_requires_explicit_semantics_assertion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    f"--convert-existing-dat={root / 'trajectory.dat'}",
                    f"--output={root / 'out.h5'}",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--assume-cbwaves-harmonic", result.stderr)

    def test_smooth_window_has_exact_endpoints_and_symmetry(self) -> None:
        time = np.asarray([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
        window = generator.smooth_window(time, 0.0, 1.0)
        derivative = generator.smooth_window_derivative(time, 0.0, 1.0)
        np.testing.assert_array_equal(window[[0, 1]], 0.0)
        np.testing.assert_array_equal(window[[-2, -1]], 1.0)
        self.assertAlmostEqual(float(window[3]), 0.5, places=15)
        np.testing.assert_allclose(window[2:5] + window[4:1:-1], 1.0)
        self.assertTrue(np.all(np.diff(window) >= 0.0))
        np.testing.assert_array_equal(derivative[[0, 1, -2, -1]], 0.0)
        self.assertAlmostEqual(float(derivative[3]), 2.0, places=15)

    def test_cubic_hermite_returns_the_analytic_derivative(self) -> None:
        source_time = np.asarray([0.0, 1.0])
        values = source_time[:, None] ** 3
        derivatives = 3.0 * source_time[:, None] ** 2
        query = np.linspace(0.0, 1.0, 17)
        interpolated, returned_derivative = generator.interpolate_cubic_hermite(
            source_time, values, derivatives, query
        )
        np.testing.assert_allclose(interpolated[:, 0], query**3, atol=2e-15)
        np.testing.assert_allclose(
            returned_derivative[:, 0], 3.0 * query**2, atol=3e-15
        )

    def test_help_and_source_free_dry_run(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--convert-existing-dat", help_result.stdout)
        self.assertIn("--force", help_result.stdout)

        dry_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(dry_result.returncode, 0, dry_result.stderr)
        plan = json.loads(dry_result.stdout)
        self.assertEqual(plan["schema"], "kpolaris.binary_trajectory.v1/1")
        self.assertEqual(plan["temporary_patch"], "9130111./3306. -> 9130111./3360.")
        self.assertEqual(plan["correction_mask"], list(generator.PN_TERMS))
        self.assertIn("checkpoint = no", plan["cbwaves_ini"])

    def test_auto_detects_archived_wrapper_dat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "official.dat"
            np.savetxt(
                path,
                np.zeros((3, len(generator.OFFICIAL_WRAPPER_DAT_COLUMNS))),
            )
            self.assertEqual(
                generator.resolve_dat_columns(path, "auto"),
                generator.OFFICIAL_WRAPPER_DAT_COLUMNS,
            )

    def test_convert_existing_dat_collapses_to_exact_remnant(self) -> None:
        mass_time = generator.CBWAVES_MSUN_TIME_S
        mass_length = generator.CBWAVES_MSUN_LENGTH_M
        time_M = np.linspace(0.0, 2.0, 9)
        phase = 0.4 * time_M
        relative_position = np.column_stack(
            (0.4 * np.cos(phase), 0.4 * np.sin(phase), np.zeros_like(phase))
        )
        relative_velocity = np.column_stack(
            (
                -0.16 * np.sin(phase),
                0.16 * np.cos(phase),
                np.zeros_like(phase),
            )
        )
        table = np.zeros((len(time_M), len(generator.DAT_COLUMNS)), dtype=np.float64)
        index = {name: column for column, name in enumerate(generator.DAT_COLUMNS)}
        table[:, index["t"]] = time_M * mass_time
        for axis, name in enumerate("xyz"):
            table[:, index[f"{name}1"]] = 0.5 * relative_position[:, axis] * mass_length
            table[:, index[f"{name}2"]] = -0.5 * relative_position[:, axis] * mass_length
            table[:, index[f"v{name}"]] = relative_velocity[:, axis]
        table[:, index["Lz"]] = 1.0
        table[:, index["r"]] = 0.4 * mass_length

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dat_path = root / "trajectory.dat"
            output = root / "trajectory.h5"
            np.savetxt(dat_path, table, fmt="%.17e")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    f"--convert-existing-dat={dat_path}",
                    "--assume-cbwaves-harmonic",
                    f"--output={output}",
                    "--merger-time-M=2.0",
                    "--transition-start-M=1.5",
                    "--transition-end-M=2.5",
                    "--post-merger-duration-M=0.5",
                    "--output-dt-M=0.125",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            summary = json.loads(result.stdout)
            self.assertEqual(summary["exact_remnant_from_M"], 2.5)

            with h5py.File(output, "r") as handle:
                reader = (SCRIPT.parents[1] / "tools/kpolaris_model_image.cpp").read_text()
                required = re.search(r'load_options.required_generator\s*=\s*"([^"]+)"', reader)
                self.assertIsNotNone(required)
                self.assertEqual(handle.attrs["generator"], required[1])
                self.assertEqual(handle["trajectory"].attrs["generator"], required[1])
                self.assertEqual(handle.attrs["schema_name"], generator.SCHEMA_NAME)
                self.assertEqual(int(handle.attrs["schema_version"]), 1)
                self.assertEqual(handle.attrs["units"], "G=c=M_ref=1")
                self.assertEqual(
                    handle["trajectory"].attrs["schema"], generator.SCHEMA_NAME
                )
                self.assertEqual(
                    handle["trajectory"].attrs["generator_version"],
                    generator.GENERATOR_VERSION,
                )
                self.assertEqual(
                    handle["trajectory"].attrs["merger_reach_contract"],
                    generator.MERGER_REACH_CONTRACT,
                )
                self.assertEqual(
                    handle["trajectory"].attrs["interpolation"],
                    "cubic_hermite_position_velocity",
                )
                self.assertEqual(
                    int(
                        handle["trajectory"].attrs[
                            "worldline_velocity_consistent"
                        ]
                    ),
                    1,
                )
                self.assertEqual(
                    handle["trajectory"].attrs["boost_velocity_model"],
                    "derivative_of_position",
                )
                self.assertEqual(int(handle.attrs["source_verified"]), 0)
                self.assertEqual(
                    int(handle.attrs["external_dat_semantics_asserted_by_user"]),
                    1,
                )
                self.assertEqual(
                    handle.attrs["trajectory_model"],
                    "external_cbwaves_dat_unverified",
                )
                self.assertEqual(int(handle.attrs["merger_separation_reached"]), 1)
                self.assertEqual(
                    int(handle["trajectory"].attrs["merger_separation_reached"]),
                    1,
                )
                self.assertEqual(
                    handle.attrs["trajectory_status"],
                    "merger_separation_reached",
                )
                self.assertIn("error below t_min", handle.attrs["out_of_range_policy"])
                self.assertIn(
                    "validated native exact-remnant tail",
                    handle.attrs["future_extension_policy"],
                )
                self.assertNotIn("source_doi", handle.attrs)
                self.assertNotIn("source_license_record", handle.attrs)
                self.assertIn(
                    "unverified external DAT", handle.attrs["source_provenance"]
                )
                expected_generator_hash = generator.sha256_file(SCRIPT)
                self.assertEqual(
                    handle.attrs["generator_sha256"], expected_generator_hash
                )
                provenance = handle["provenance"].attrs
                self.assertEqual(provenance["generator_sha256"], expected_generator_hash)
                self.assertNotIn("remnant_fit_version", provenance)
                self.assertNotIn(
                    "remnant_fit_reference_precession_sha256", provenance
                )
                self.assertEqual(
                    provenance["source_verification"], "unverified-external-dat"
                )
                for name in ("python_version", "numpy_version", "h5py_version"):
                    self.assertTrue(provenance[name])
                self.assertEqual(
                    int(handle.attrs["worldline_velocity_consistent"]), 1
                )
                time = handle["time"][...]
                window = handle["window"][...]
                position = handle["position"][...]
                velocity = handle["velocity"][...]
                mass = handle["mass"][...]
                spin_a = handle["spin_a"][...]
                np.testing.assert_array_equal(
                    velocity,
                    handle["diagnostics/position_time_derivative"][...],
                )
                exact = time >= 2.5
                self.assertTrue(np.any(time == 1.5))
                self.assertTrue(np.any(time == 2.5))
                np.testing.assert_array_equal(window[time <= 1.5], 0.0)
                np.testing.assert_array_equal(window[exact], 1.0)
                np.testing.assert_allclose(position[exact, 0], position[exact, 1], atol=0.0)
                np.testing.assert_allclose(velocity[exact], 0.0, atol=0.0)
                np.testing.assert_allclose(mass[exact], 0.95173 / 2.0, rtol=0.0, atol=1e-15)
                np.testing.assert_allclose(spin_a[exact, 0], spin_a[exact, 1], atol=0.0)
                np.testing.assert_allclose(
                    spin_a[exact, 0],
                    np.broadcast_to(
                        np.asarray([0.0, 0.0, 0.95173 * 0.68646]),
                        spin_a[exact, 0].shape,
                    ),
                    rtol=0.0,
                    atol=1e-15,
                )
                self.assertEqual(
                    handle["provenance"].attrs["patch"],
                    "not-asserted; no source was supplied or modified",
                )

            original = output.read_bytes()
            blocked = subprocess.run(
                result.args, text=True, capture_output=True, check=False
            )
            self.assertEqual(blocked.returncode, 2)
            self.assertIn("output already exists", blocked.stderr)
            self.assertEqual(output.read_bytes(), original)

            forced = subprocess.run(
                [*result.args, "--force"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertTrue(output.is_file())

    def test_paper_nr_fit_writes_deterministic_full_provenance(self) -> None:
        time_M = np.linspace(0.0, 2.0, 9)
        phase = 0.3 * time_M
        relative_position = np.column_stack(
            (0.6 * np.cos(phase), 0.6 * np.sin(phase), np.zeros_like(phase))
        )
        relative_velocity = np.column_stack(
            (-0.18 * np.sin(phase), 0.18 * np.cos(phase), np.zeros_like(phase))
        )
        mass1 = 2.0 / 3.0
        mass2 = 1.0 / 3.0
        theta1 = np.arccos(0.3)
        theta2 = np.arccos(-0.4)
        delta_phi = 1.2
        spin1 = 0.7 * np.asarray([np.sin(theta1), 0.0, np.cos(theta1)])
        spin2 = 0.4 * np.asarray(
            [
                np.sin(theta2) * np.cos(delta_phi),
                np.sin(theta2) * np.sin(delta_phi),
                np.cos(theta2),
            ]
        )
        table = np.zeros((len(time_M), len(generator.DAT_COLUMNS)), dtype=np.float64)
        index = {name: column for column, name in enumerate(generator.DAT_COLUMNS)}
        table[:, index["t"]] = time_M * generator.CBWAVES_MSUN_TIME_S
        for axis_number, name in enumerate("xyz"):
            table[:, index[f"{name}1"]] = (
                mass2 * relative_position[:, axis_number]
                * generator.CBWAVES_MSUN_LENGTH_M
            )
            table[:, index[f"{name}2"]] = (
                -mass1 * relative_position[:, axis_number]
                * generator.CBWAVES_MSUN_LENGTH_M
            )
            table[:, index[f"v{name}"]] = relative_velocity[:, axis_number]
            table[:, index[f"s1{name}"]] = spin1[axis_number]
            table[:, index[f"s2{name}"]] = spin2[axis_number]
        table[:, index["Lz"]] = 1.0
        table[:, index["r"]] = 0.6 * generator.CBWAVES_MSUN_LENGTH_M

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dat_path = root / "trajectory.dat"
            output = root / "paper_nr_fit.h5"
            np.savetxt(dat_path, table, fmt="%.17e")
            command = [
                sys.executable,
                str(SCRIPT),
                f"--convert-existing-dat={dat_path}",
                "--assume-cbwaves-harmonic",
                f"--output={output}",
                f"--mass1-fraction={mass1:.17g}",
                f"--mass2-fraction={mass2:.17g}",
                "--merger-time-M=2.0",
                "--transition-start-M=1.5",
                "--transition-end-M=2.5",
                "--post-merger-duration-M=0.25",
                "--output-dt-M=0.125",
                "--remnant-model=paper-nr-fit",
                "--remnant-fit-time-M=1.75",
                "--remnant-kick-phase-rad=0.37",
            ]
            result = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["remnant_model"], "paper-nr-fit")
            self.assertTrue(np.all(np.isfinite(summary["remnant_kick"])))

            with h5py.File(output, "r") as handle:
                self.assertEqual(handle.attrs["remnant_model"], "paper-nr-fit")
                self.assertEqual(handle["remnant"].attrs["model"], "paper-nr-fit")
                fit = handle["remnant/fit"]
                self.assertEqual(
                    fit.attrs["version"], generator.REMNANT_FIT_VERSION
                )
                self.assertEqual(
                    fit.attrs["reference_precession_sha256"],
                    generator.PRECESSION_REFERENCE_SHA256,
                )
                self.assertEqual(
                    fit.attrs["kick_phase_policy"],
                    "explicit-radians-no-random-sampling",
                )
                self.assertIn("all enabled", fit.attrs["kick_terms"])
                self.assertEqual(
                    fit.attrs["kick_zero_spin_policy"],
                    "analytic continuous zero-spin limit",
                )
                self.assertAlmostEqual(float(fit.attrs["kick_phase_rad"]), 0.37)
                self.assertEqual(int(fit.attrs["sample_index"]), 7)
                self.assertAlmostEqual(float(fit.attrs["sample_time"]), 1.75)
                np.testing.assert_allclose(fit["spin_chi"][...], [spin1, spin2])
                basis = fit["orbital_basis"][...]
                np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=2e-15)
                np.testing.assert_allclose(np.linalg.det(basis), 1.0, atol=2e-15)
                np.testing.assert_allclose(
                    handle["remnant/kick"][...],
                    fit["kick_components_orbital"][...] @ basis,
                    atol=2e-15,
                )
                self.assertAlmostEqual(
                    float(handle["remnant/mass"][...]),
                    0.9560013454998099,
                    places=14,
                )
                self.assertAlmostEqual(
                    float(np.linalg.norm(handle["remnant/spin_chi"][...])),
                    0.7452111456013830,
                    places=14,
                )
                exact = handle["window"][...] == 1.0
                expected_half_mass = 0.5 * float(handle["remnant/mass"][...])
                np.testing.assert_allclose(
                    handle["mass"][...][exact], expected_half_mass, atol=2e-15
                )
                provenance = handle["provenance"].attrs
                self.assertEqual(
                    provenance["remnant_fit_reference_precession_sha256"],
                    generator.PRECESSION_REFERENCE_SHA256,
                )

            default_output = root / "paper_nr_fit_default_time.h5"
            default_command = [
                (
                    f"--output={default_output}"
                    if item.startswith("--output=")
                    else item
                )
                for item in command
                if not item.startswith("--remnant-fit-time-M=")
            ]
            default_command.append("--transition-velocity=paper-blend")
            default_result = subprocess.run(
                default_command, text=True, capture_output=True, check=False
            )
            self.assertEqual(default_result.returncode, 0, default_result.stderr)
            with h5py.File(default_output, "r") as handle:
                fit = handle["remnant/fit"].attrs
                self.assertAlmostEqual(float(fit["requested_time"]), 1.5)
                self.assertAlmostEqual(float(fit["sample_time"]), 1.5)
                self.assertEqual(
                    fit["requested_time_policy"],
                    "actual-transition-start-paper-prescription",
                )
                self.assertEqual(
                    handle["trajectory"].attrs["interpolation"], "linear"
                )

    def test_explicit_remnant_is_required_for_generic_binary(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--dry-run",
                "--mass1-fraction=0.6",
                "--mass2-fraction=0.4",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--remnant-model=explicit", result.stderr)

    def test_transition_speed_guard_auto_expansion_and_paper_mode(self) -> None:
        time_M = np.linspace(0.0, 4.0, 41)
        table = np.zeros((len(time_M), len(generator.DAT_COLUMNS)), dtype=np.float64)
        index = {name: column for column, name in enumerate(generator.DAT_COLUMNS)}
        table[:, index["t"]] = time_M * generator.CBWAVES_MSUN_TIME_S
        table[:, index["x1"]] = 0.2 * generator.CBWAVES_MSUN_LENGTH_M
        table[:, index["x2"]] = -0.2 * generator.CBWAVES_MSUN_LENGTH_M
        table[:, index["Lz"]] = 1.0
        table[:, index["r"]] = 0.4 * generator.CBWAVES_MSUN_LENGTH_M

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dat_path = root / "trajectory.dat"
            np.savetxt(dat_path, table, fmt="%.17e")
            truncated_output = root / "truncated.h5"
            truncated = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    f"--convert-existing-dat={dat_path}",
                    "--assume-cbwaves-harmonic",
                    f"--output={truncated_output}",
                    "--rmin-M=0.1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(truncated.returncode, 2)
            self.assertIn("selected merger state", truncated.stderr)
            diagnostic_output = root / "diagnostic_truncated.h5"
            diagnostic = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    f"--convert-existing-dat={dat_path}",
                    "--assume-cbwaves-harmonic",
                    f"--output={diagnostic_output}",
                    "--rmin-M=0.1",
                    "--allow-truncated-inspiral",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(diagnostic.returncode, 0, diagnostic.stderr)
            diagnostic_summary = json.loads(diagnostic.stdout)
            self.assertFalse(diagnostic_summary["merger_separation_reached"])
            self.assertEqual(
                diagnostic_summary["trajectory_status"], "diagnostic_truncated"
            )
            with h5py.File(diagnostic_output, "r") as handle:
                self.assertEqual(int(handle.attrs["merger_separation_reached"]), 0)
                self.assertEqual(
                    int(handle["trajectory"].attrs["merger_separation_reached"]),
                    0,
                )
                self.assertEqual(
                    handle.attrs["trajectory_status"], "diagnostic_truncated"
                )
                self.assertAlmostEqual(float(handle.attrs["final_pn_separation"]), 0.4)
            common = [
                sys.executable,
                str(SCRIPT),
                f"--convert-existing-dat={dat_path}",
                "--assume-cbwaves-harmonic",
                "--merger-time-M=4.0",
                "--post-merger-duration-M=0.1",
                "--output-dt-M=0.005",
            ]

            auto_output = root / "auto.h5"
            auto = subprocess.run(
                common + [f"--output={auto_output}"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(auto.returncode, 0, auto.stderr)
            auto_summary = json.loads(auto.stdout)
            self.assertGreater(auto_summary["transition_auto_expansions"], 0)
            self.assertLessEqual(auto_summary["maximum_boost_speed"], 0.95)

            narrow = [
                "--transition-start-M=3.99",
                "--transition-end-M=4.01",
            ]
            rejected_output = root / "rejected.h5"
            rejected = subprocess.run(
                common + narrow + [f"--output={rejected_output}"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("worldline-consistent merger transition", rejected.stderr)

            paper_output = root / "paper.h5"
            paper = subprocess.run(
                common
                + narrow
                + ["--transition-velocity=paper-blend", f"--output={paper_output}"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(paper.returncode, 0, paper.stderr)
            with h5py.File(paper_output, "r") as handle:
                self.assertEqual(
                    int(handle.attrs["worldline_velocity_consistent"]), 0
                )
                self.assertEqual(
                    handle.attrs["boost_velocity_model"],
                    "paper_eq16_independent_blend",
                )
                np.testing.assert_array_equal(handle["velocity"][...], 0.0)
                self.assertGreater(
                    float(
                        np.max(
                            np.abs(
                                handle[
                                    "diagnostics/position_time_derivative"
                                ][...]
                            )
                        )
                    ),
                    1.0,
                )


if __name__ == "__main__":
    unittest.main()
