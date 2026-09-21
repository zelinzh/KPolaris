#!/usr/bin/env python3
"""Relate per-pixel double-pass diagnostics to Stokes image differences."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from compare_stokes_hdf5 import compare, load_image


DIAGNOSTICS = (
    "closure_x",
    "closure_k",
    "final_null",
    "frame_error",
    "basis_identity_error",
)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average zero-based ranks without requiring SciPy."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    begin = 0
    while begin < values.size:
        end = begin + 1
        while end < values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        begin = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def load_diagnostics(path: Path, frame_index: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    group_name = f"frame_{frame_index}/diagnostics"
    with h5py.File(path, "r") as h5:
        if group_name not in h5:
            raise ValueError(f"{path}: missing {group_name}")
        group = h5[group_name]
        arrays: dict[str, np.ndarray] = {}
        for name in DIAGNOSTICS:
            if name not in group:
                raise ValueError(f"{path}: missing {group_name}/{name}")
            array = np.abs(np.asarray(group[name][...], dtype=np.float64))
            if not np.isfinite(array).all():
                raise ValueError(f"{path}: {name} contains NaN or Inf")
            arrays[name] = array
        metadata = {
            "adaptive_tolerance": float(h5.attrs.get("adaptive_tolerance", math.nan)),
            "mean_total_steps": float(group.attrs.get(
                "mean_total_steps", h5.attrs.get("mean_total_steps", math.nan))),
        }
    return metadata, arrays


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def analyze_candidate(
    reference_path: Path,
    candidate_path: Path,
    frame_index: int,
    frequency_index: int,
    min_reference_i_fraction: float,
) -> dict[str, object]:
    stokes_names = ["I", "Q", "U", "V"]
    reference_meta, reference = load_image(
        reference_path, frame_index, frequency_index, stokes_names)
    candidate_meta, candidate = load_image(
        candidate_path, frame_index, frequency_index, stokes_names)
    if reference_meta["shape"] != candidate_meta["shape"]:
        raise ValueError(f"{candidate_path}: image shape differs from reference")
    metrics = compare(reference, candidate)

    delta_stack = np.stack(
        [candidate[name] - reference[name] for name in stokes_names], axis=0)
    reference_stack = np.stack([reference[name] for name in stokes_names], axis=0)
    pixel_error = np.sqrt(np.sum(delta_stack * delta_stack, axis=0))
    reference_peak = float(np.max(np.sqrt(np.sum(reference_stack * reference_stack, axis=0))))
    normalized_pixel_error = pixel_error / reference_peak if reference_peak > 0.0 else pixel_error

    metadata, diagnostics = load_diagnostics(candidate_path, frame_index)
    reference_i_peak = float(np.max(np.abs(reference["I"])))
    emitting_mask = (np.abs(reference["I"]) >=
                     min_reference_i_fraction * reference_i_peak)
    if not np.any(emitting_mask):
        raise ValueError("emitting-pixel mask is empty; lower --min-reference-i-fraction")
    flat_error = normalized_pixel_error[emitting_mask].ravel()
    correlations: dict[str, dict[str, float | None]] = {}
    diagnostic_quantiles: dict[str, dict[str, float]] = {}
    diagnostic_quantiles_all: dict[str, dict[str, float]] = {}
    positive_error = flat_error[flat_error > 0.0]
    error_floor = (float(np.min(positive_error)) * 0.5
                   if positive_error.size else np.finfo(np.float64).tiny)
    log_error = np.log10(flat_error + error_floor)
    for name, array in diagnostics.items():
        diagnostic_quantiles_all[name] = quantiles(array.ravel())
        flat = array[emitting_mask].ravel()
        positive = flat[flat > 0.0]
        floor = (float(np.min(positive)) * 0.5
                 if positive.size else np.finfo(np.float64).tiny)
        log_diagnostic = np.log10(flat + floor)
        correlations[name] = {
            "log10_pearson": correlation(log_diagnostic, log_error),
            "spearman": correlation(rankdata(flat), rankdata(flat_error)),
        }
        diagnostic_quantiles[name] = quantiles(flat)

    worst_l1 = max(row["l1_relative"] for row in metrics.values())
    worst_linf = max(row["linf_over_peak"] for row in metrics.values())
    return {
        "candidate": str(candidate_path),
        "adaptive_tolerance": metadata["adaptive_tolerance"],
        "shape": reference_meta["shape"],
        "metrics": metrics,
        "worst_relative_l1": worst_l1,
        "worst_linf_over_peak": worst_linf,
        "min_reference_i_fraction": min_reference_i_fraction,
        "emitting_pixel_count": int(np.count_nonzero(emitting_mask)),
        "emitting_pixel_fraction": float(np.mean(emitting_mask)),
        "pixel_stokes_error_over_reference_peak": quantiles(flat_error),
        "diagnostics_emitting_pixels": diagnostic_quantiles,
        "diagnostics_all_pixels": diagnostic_quantiles_all,
        "correlations_with_pixel_stokes_error": correlations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--frequency-index", type=int, default=0)
    parser.add_argument("--min-reference-i-fraction", type=float, default=1e-6)
    parser.add_argument("--json-output", required=True, type=Path)
    args = parser.parse_args()

    candidate_rows = [
        analyze_candidate(
            args.reference, path, args.frame_index, args.frequency_index,
            args.min_reference_i_fraction)
        for path in args.candidates
    ]
    sweep_correlations: dict[str, float | None] = {}
    sweep_error = np.asarray(
        [row["worst_relative_l1"] for row in candidate_rows], dtype=np.float64)
    for name in DIAGNOSTICS:
        values = np.asarray([
            row["diagnostics_emitting_pixels"][name]["p99"]
            for row in candidate_rows
        ], dtype=np.float64)
        if np.all(values > 0.0) and np.all(sweep_error > 0.0):
            sweep_correlations[name] = correlation(
                np.log10(values), np.log10(sweep_error))
        else:
            sweep_correlations[name] = None

    payload = {
        "schema": "kpolaris_closure_stokes_analysis",
        "schema_version": 1,
        "reference": str(args.reference),
        "frame_index": args.frame_index,
        "frequency_index": args.frequency_index,
        "candidates": candidate_rows,
        "sweep_log10_pearson_p99_diagnostic_vs_worst_l1": sweep_correlations,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("tolerance worst_L1 worst_Linf closure_x_p99 closure_k_p99 "
          "frame_p99 basis_p99 closure_x_spearman")
    for row in payload["candidates"]:
        diagnostics = row["diagnostics_emitting_pixels"]
        correlations = row["correlations_with_pixel_stokes_error"]
        print(
            f"{row['adaptive_tolerance']:.1e} "
            f"{row['worst_relative_l1']:.6e} "
            f"{row['worst_linf_over_peak']:.6e} "
            f"{diagnostics['closure_x']['p99']:.6e} "
            f"{diagnostics['closure_k']['p99']:.6e} "
            f"{diagnostics['frame_error']['p99']:.6e} "
            f"{diagnostics['basis_identity_error']['p99']:.6e} "
            f"{correlations['closure_x']['spearman']!s}"
        )
    print("sweep_log10_pearson_p99_diagnostic_vs_worst_l1 " +
          json.dumps(sweep_correlations, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
