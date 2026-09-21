#!/usr/bin/env python3
"""Compare two KPolaris HDF5 Stokes images with publication-style norms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from kpolaris_evpa import evpa_zero, qu_basis_sign


def text_value(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def finite_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else math.inf


def load_image(
    path: Path,
    frame_index: int,
    frequency_index: int,
    stokes: list[str],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with h5py.File(path, "r") as h5:
        if text_value(h5.attrs.get("schema", "")) != "kpolaris_image":
            raise ValueError(f"{path}: expected schema='kpolaris_image'")
        group_name = f"frame_{frame_index}/freq_{frequency_index}"
        if group_name not in h5:
            raise ValueError(f"{path}: missing {group_name}")
        group = h5[group_name]
        source_zero = evpa_zero(h5.attrs, group.attrs)
        arrays: dict[str, np.ndarray] = {}
        for name in stokes:
            if name not in group:
                raise ValueError(f"{path}: missing {group_name}/{name}")
            array = np.asarray(group[name][...], dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"{path}: {group_name}/{name} contains NaN or Inf")
            if name in ("Q", "U"):
                array = array * qu_basis_sign(source_zero, "N")
            arrays[name] = array
        metadata: dict[str, object] = {
            "schema_version": int(h5.attrs.get("schema_version", 0)),
            "input_evpa_0": source_zero,
            "comparison_evpa_0": "N",
            "model": text_value(h5.attrs.get("model", "")),
            "coordinate": text_value(h5.attrs.get("coordinate", "")),
            "frequency_hz": float(group.attrs.get(
                "frequency_hz", h5.attrs.get("frequency_hz", math.nan))),
            "shape": list(next(iter(arrays.values())).shape),
        }
        if "grid/x" in h5 and "grid/y" in h5:
            metadata["grid_x"] = np.asarray(h5["grid/x"][...], dtype=np.float64)
            metadata["grid_y"] = np.asarray(h5["grid/y"][...], dtype=np.float64)
        return metadata, arrays


def compare(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, ref in reference.items():
        got = candidate[name]
        if got.shape != ref.shape:
            raise ValueError(
                f"{name}: shape mismatch, reference={ref.shape}, candidate={got.shape}")
        delta = got - ref
        ref_sum = float(np.sum(ref))
        got_sum = float(np.sum(got))
        results[name] = {
            "reference_sum": ref_sum,
            "candidate_sum": got_sum,
            "sum_relative": ((got_sum - ref_sum) / ref_sum
                             if ref_sum != 0.0
                             else (0.0 if got_sum == 0.0 else math.inf)),
            "l1_relative": finite_ratio(
                float(np.sum(np.abs(delta))), float(np.sum(np.abs(ref)))),
            "rms_relative": math.sqrt(finite_ratio(
                float(np.sum(delta * delta)), float(np.sum(ref * ref)))),
            "linf_over_peak": finite_ratio(
                float(np.max(np.abs(delta))), float(np.max(np.abs(ref)))),
        }
    return results


def load_diagnostics(path: Path, frame_index: int) -> dict[str, np.ndarray]:
    group_name = f"frame_{frame_index}/diagnostics"
    names = (
        "reason", "pass_a_steps", "steps", "total_steps", "closure_x",
        "closure_k", "final_null", "frame_error", "basis_identity_error",
    )
    with h5py.File(path, "r") as h5:
        if group_name not in h5:
            raise ValueError(f"{path}: missing {group_name}")
        group = h5[group_name]
        source_zero = evpa_zero(h5.attrs, group.attrs)
        arrays: dict[str, np.ndarray] = {}
        for name in names:
            if name not in group:
                raise ValueError(f"{path}: missing {group_name}/{name}")
            arrays[name] = np.asarray(group[name][...])
        return arrays


def compare_diagnostics(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    exact_names = {"reason", "pass_a_steps", "steps", "total_steps"}
    for name, ref in reference.items():
        got = candidate[name]
        if got.shape != ref.shape:
            raise ValueError(
                f"diagnostic {name}: shape mismatch, "
                f"reference={ref.shape}, candidate={got.shape}")
        if name in exact_names:
            delta = got.astype(np.int64) - ref.astype(np.int64)
            results[name] = {
                "mismatched_values": int(np.count_nonzero(delta)),
                "max_absolute_difference": int(np.max(np.abs(delta))),
            }
            continue
        ref_float = ref.astype(np.float64)
        got_float = got.astype(np.float64)
        if not np.isfinite(ref_float).all() or not np.isfinite(got_float).all():
            raise ValueError(f"diagnostic {name} contains NaN or Inf")
        delta = got_float - ref_float
        results[name] = {
            "l1_relative": finite_ratio(
                float(np.sum(np.abs(delta))), float(np.sum(np.abs(ref_float)))),
            "linf": float(np.max(np.abs(delta))),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--frequency-index", type=int, default=0)
    parser.add_argument("--stokes", default="I,Q,U,V")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--rtol-l1", type=float)
    parser.add_argument("--rtol-linf-peak", type=float)
    parser.add_argument("--compare-diagnostics", action="store_true",
                        help="also compare termination, step-count, and closure arrays")
    parser.add_argument("--require-exact-discrete-diagnostics", action="store_true",
                        help="fail if reason or Pass A/B/total step arrays differ")
    parser.add_argument("--require-exact-reason", action="store_true",
                        help="fail if any ray-termination reason differs")
    parser.add_argument("--max-step-mismatch-fraction", type=float,
                        help="maximum allowed differing-pixel fraction for each step array")
    parser.add_argument("--max-step-absolute-difference", type=int,
                        help="maximum allowed absolute difference in each step array")
    args = parser.parse_args()

    stokes = [item.strip() for item in args.stokes.split(",") if item.strip()]
    if not stokes:
        parser.error("--stokes must contain at least one dataset name")

    reference_meta, reference = load_image(
        args.reference, args.frame_index, args.frequency_index, stokes)
    candidate_meta, candidate = load_image(
        args.candidate, args.frame_index, args.frequency_index, stokes)

    grid_comparison: dict[str, dict[str, float]] = {}
    for key in ("grid_x", "grid_y"):
        ref_grid = reference_meta.pop(key, None)
        got_grid = candidate_meta.pop(key, None)
        if (ref_grid is None) != (got_grid is None):
            raise ValueError(f"grid mismatch: only one file contains {key}")
        if ref_grid is not None:
            max_abs = float(np.max(np.abs(ref_grid - got_grid)))
            scale = float(max(np.max(np.abs(ref_grid)), np.max(np.abs(got_grid))))
            tolerance = 64.0 * np.finfo(np.float64).eps * scale
            grid_comparison[key] = {
                "max_absolute_difference": max_abs,
                "scale": scale,
                "absolute_tolerance": tolerance,
            }
            if max_abs > tolerance:
                raise ValueError(
                    f"grid mismatch in {key}: max_abs={max_abs:.17e}, "
                    f"tolerance={tolerance:.17e}")
    if reference_meta["shape"] != candidate_meta["shape"]:
        raise ValueError(
            f"image shape mismatch: {reference_meta['shape']} != {candidate_meta['shape']}")
    ref_frequency = float(reference_meta["frequency_hz"])
    got_frequency = float(candidate_meta["frequency_hz"])
    if not math.isclose(ref_frequency, got_frequency, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(
            f"frequency mismatch: {ref_frequency:.17g} != {got_frequency:.17g}")

    results = compare(reference, candidate)
    diagnostic_results: dict[str, dict[str, float | int]] = {}
    if args.compare_diagnostics:
        diagnostic_results = compare_diagnostics(
            load_diagnostics(args.reference, args.frame_index),
            load_diagnostics(args.candidate, args.frame_index))
    print(f"reference: {args.reference}")
    print(f"candidate: {args.candidate}")
    print(f"shape: {reference_meta['shape']}")
    print(f"frequency_hz: {ref_frequency:.17g}")
    print("stokes sum_relative l1_relative rms_relative linf_over_peak")
    for name in stokes:
        row = results[name]
        print(
            f"{name} {row['sum_relative']:.17e} {row['l1_relative']:.17e} "
            f"{row['rms_relative']:.17e} {row['linf_over_peak']:.17e}")

    worst_l1 = max(row["l1_relative"] for row in results.values())
    worst_linf = max(row["linf_over_peak"] for row in results.values())
    print(f"worst_rel_L1={worst_l1:.17e}")
    print(f"worst_Linf_over_peak={worst_linf:.17e}")
    for key, row in grid_comparison.items():
        print(
            f"{key}_max_abs={row['max_absolute_difference']:.17e} "
            f"tolerance={row['absolute_tolerance']:.17e}")
    if diagnostic_results:
        print("diagnostic metric_1 metric_2")
        for name, row in diagnostic_results.items():
            if "mismatched_values" in row:
                print(
                    f"{name} mismatched={row['mismatched_values']} "
                    f"max_abs={row['max_absolute_difference']}")
            else:
                print(
                    f"{name} l1_relative={row['l1_relative']:.17e} "
                    f"linf={row['linf']:.17e}")

    if args.json_output is not None:
        payload = {
            "reference": str(args.reference),
            "candidate": str(args.candidate),
            "frame_index": args.frame_index,
            "frequency_index": args.frequency_index,
            "reference_metadata": reference_meta,
            "candidate_metadata": candidate_meta,
            "metrics": results,
            "grid_comparison": grid_comparison,
            "diagnostics": diagnostic_results,
            "worst_relative_l1": worst_l1,
            "worst_linf_over_peak": worst_linf,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.rtol_l1 is not None and worst_l1 > args.rtol_l1:
        return 2
    if args.rtol_linf_peak is not None and worst_linf > args.rtol_linf_peak:
        return 3
    if args.require_exact_discrete_diagnostics:
        if not diagnostic_results:
            parser.error(
                "--require-exact-discrete-diagnostics requires --compare-diagnostics")
        if any(int(diagnostic_results[name]["mismatched_values"]) != 0
               for name in ("reason", "pass_a_steps", "steps", "total_steps")):
            return 4
    if args.require_exact_reason:
        if not diagnostic_results:
            parser.error("--require-exact-reason requires --compare-diagnostics")
        if int(diagnostic_results["reason"]["mismatched_values"]) != 0:
            return 4
    step_names = ("pass_a_steps", "steps", "total_steps")
    if args.max_step_mismatch_fraction is not None:
        if not diagnostic_results:
            parser.error(
                "--max-step-mismatch-fraction requires --compare-diagnostics")
        if args.max_step_mismatch_fraction < 0.0:
            parser.error("--max-step-mismatch-fraction must be nonnegative")
        pixel_count = int(reference_meta["shape"][0] * reference_meta["shape"][1])
        if any((int(diagnostic_results[name]["mismatched_values"]) / pixel_count)
               > args.max_step_mismatch_fraction for name in step_names):
            return 5
    if args.max_step_absolute_difference is not None:
        if not diagnostic_results:
            parser.error(
                "--max-step-absolute-difference requires --compare-diagnostics")
        if args.max_step_absolute_difference < 0:
            parser.error("--max-step-absolute-difference must be nonnegative")
        if any(int(diagnostic_results[name]["max_absolute_difference"])
               > args.max_step_absolute_difference for name in step_names):
            return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
