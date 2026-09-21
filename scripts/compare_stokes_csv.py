#!/usr/bin/env python3
"""Compare two KPolaris-style Stokes CSV image files.

The script intentionally uses only the Python standard library. It compares rows
by (ix, iy) and reports L1/Linf errors plus reference-normalized NMSE
and NRMSE (Prather et al. 2023, equation 19) for selected Stokes columns.
Comment metadata lines of the form "# key,value" are ignored for the numeric
comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def read_image(path: Path) -> tuple[dict[str, str], dict[tuple[int, int], dict[str, float]]]:
    metadata: dict[str, str] = {}
    rows: dict[tuple[int, int], dict[str, float]] = {}
    header: list[str] | None = None

    with path.open(newline="") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if "," in body:
                    key, value = body.split(",", 1)
                    metadata[key.strip()] = value.strip()
                continue
            if header is None:
                header = next(csv.reader([line]))
                continue
            values = next(csv.reader([line]))
            if len(values) != len(header):
                raise ValueError(f"{path}: malformed row with {len(values)} fields, expected {len(header)}")
            record = dict(zip(header, values))
            key = (int(record["ix"]), int(record["iy"]))
            rows[key] = {name: float(value) for name, value in record.items() if name not in {"ix", "iy"}}

    if header is None:
        raise ValueError(f"{path}: no CSV header found")
    return metadata, rows


def compare_columns(ref: dict[tuple[int, int], dict[str, float]],
                    got: dict[tuple[int, int], dict[str, float]],
                    columns: Iterable[str],
                    ignore_nonfinite: bool) -> tuple[
                        list[str], dict[str, dict[str, object]], float, float,
                        int, int]:
    if ref.keys() != got.keys():
        missing = sorted(ref.keys() - got.keys())[:5]
        extra = sorted(got.keys() - ref.keys())[:5]
        raise ValueError(f"pixel sets differ; missing={missing}, extra={extra}")

    lines: list[str] = []
    worst_abs = 0.0
    worst_rel_l1 = 0.0
    skipped_pixels: set[tuple[int, int]] = set()
    used_values = 0
    metrics: dict[str, dict[str, object]] = {}
    for col in columns:
        abs_l1 = 0.0
        ref_l1 = 0.0
        ref_l2 = 0.0
        diff_l2 = 0.0
        linf = 0.0
        linf_pixel = None
        used_col = 0
        skipped_col = 0
        for key in sorted(ref):
            ref_value = ref[key][col]
            got_value = got[key][col]
            if not (math.isfinite(ref_value) and math.isfinite(got_value)):
                if ignore_nonfinite:
                    skipped_pixels.add(key)
                    skipped_col += 1
                    continue
                raise ValueError(f"non-finite value in {col} at pixel {key}: reference={ref_value}, candidate={got_value}")
            diff = abs(got_value - ref_value)
            abs_l1 += diff
            ref_l1 += abs(ref_value)
            # Accumulate norms without squaring dimensional intensities, which
            # can underflow for invariant Stokes or overflow for large units.
            ref_l2 = math.hypot(ref_l2, ref_value)
            diff_l2 = math.hypot(diff_l2, diff)
            if diff > linf:
                linf = diff
                linf_pixel = key
            used_col += 1
        rel_l1 = abs_l1 / ref_l1 if ref_l1 > 0 else (0.0 if abs_l1 == 0 else math.inf)
        nrmse = (diff_l2 / ref_l2 if ref_l2 > 0
                 else (0.0 if diff_l2 == 0 else math.inf))
        nmse = nrmse * nrmse
        worst_abs = max(worst_abs, linf)
        worst_rel_l1 = max(worst_rel_l1, rel_l1)
        used_values += used_col
        suffix = f" used={used_col}"
        if skipped_col:
            suffix += f" skipped_nonfinite={skipped_col}"
        lines.append(
            f"{col}: L1_abs={abs_l1:.17e} L1_rel={rel_l1:.17e} "
            f"Linf={linf:.17e} NMSE={nmse:.17e} NRMSE={nrmse:.17e} "
            f"pixel={linf_pixel}{suffix}")
        metrics[col] = {
            "absolute_l1": abs_l1,
            "relative_l1": rel_l1,
            "nmse": nmse,
            "nrmse": nrmse,
            "linf": linf,
            "linf_pixel": list(linf_pixel) if linf_pixel is not None else None,
            "used_values": used_col,
            "skipped_nonfinite": skipped_col,
        }
    return (lines, metrics, worst_abs, worst_rel_l1, len(skipped_pixels),
            used_values)


def integrated_stokes(image: dict[tuple[int, int], dict[str, float]],
                      suffix: str, ignore_nonfinite: bool) -> dict[str, object]:
    names = {stokes: f"{stokes}_{suffix}" for stokes in "IQUV"}
    sums = {stokes: 0.0 for stokes in "IQUV"}
    used = 0
    skipped = 0
    for key in sorted(image):
        values = {stokes: image[key][name] for stokes, name in names.items()}
        if not all(math.isfinite(value) for value in values.values()):
            if ignore_nonfinite:
                skipped += 1
                continue
            raise ValueError(
                f"non-finite integrated Stokes value at pixel {key}: {values}")
        for stokes, value in values.items():
            sums[stokes] += value
        used += 1

    intensity = sums["I"]
    lp_fraction = (math.hypot(sums["Q"], sums["U"]) / intensity
                   if intensity != 0.0 else math.nan)
    cp_fraction = sums["V"] / intensity if intensity != 0.0 else math.nan
    evpa_deg = 0.5 * math.degrees(math.atan2(sums["U"], sums["Q"]))
    return {
        "stokes_sums": sums,
        "linear_polarization_fraction": lp_fraction,
        "circular_polarization_fraction": cp_fraction,
        "evpa_deg": evpa_deg,
        "used_pixels": used,
        "skipped_nonfinite_pixels": skipped,
    }


def wrapped_evpa_difference(candidate_deg: float, reference_deg: float) -> float:
    difference = candidate_deg - reference_deg
    while difference > 90.0:
        difference -= 180.0
    while difference <= -90.0:
        difference += 180.0
    return difference


def integrated_comparison(
        ref: dict[tuple[int, int], dict[str, float]],
        got: dict[tuple[int, int], dict[str, float]],
        ignore_nonfinite: bool) -> dict[str, dict[str, object]]:
    if not ref:
        return {}
    available = next(iter(ref.values())).keys()
    result: dict[str, dict[str, object]] = {}
    for suffix in ("nu", "inv"):
        if not all(f"{stokes}_{suffix}" in available for stokes in "IQUV"):
            continue
        reference = integrated_stokes(ref, suffix, ignore_nonfinite)
        candidate = integrated_stokes(got, suffix, ignore_nonfinite)
        ref_sums = reference["stokes_sums"]
        got_sums = candidate["stokes_sums"]
        assert isinstance(ref_sums, dict) and isinstance(got_sums, dict)
        relative_differences = {
            stokes: ((got_sums[stokes] - ref_sums[stokes]) / ref_sums[stokes]
                     if ref_sums[stokes] != 0.0 else math.nan)
            for stokes in "IQUV"
        }
        result[suffix] = {
            "reference": reference,
            "candidate": candidate,
            "candidate_minus_reference": {
                "stokes_sum_relative": relative_differences,
                "linear_polarization_fraction_absolute": (
                    candidate["linear_polarization_fraction"]
                    - reference["linear_polarization_fraction"]),
                "circular_polarization_fraction_absolute": (
                    candidate["circular_polarization_fraction"]
                    - reference["circular_polarization_fraction"]),
                "evpa_deg_wrapped": wrapped_evpa_difference(
                    float(candidate["evpa_deg"]), float(reference["evpa_deg"])),
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--columns", default="I_inv,Q_inv,U_inv,V_inv,I_nu,Q_nu,U_nu,V_nu",
                        help="comma-separated columns to compare")
    parser.add_argument("--atol", type=float, default=None,
                        help="fail if any selected column Linf exceeds this")
    parser.add_argument("--rtol-l1", type=float, default=None,
                        help="fail if any selected column relative L1 exceeds this")
    parser.add_argument("--ignore-nonfinite", action="store_true",
                        help="skip pixels where either image has NaN or Inf in a selected column")
    parser.add_argument("--json-output", type=Path,
                        help="write metadata and all comparison metrics as JSON")
    args = parser.parse_args()

    ref_meta, ref = read_image(args.reference)
    got_meta, got = read_image(args.candidate)
    columns = [item.strip() for item in args.columns.split(",") if item.strip()]
    (lines, metrics, worst_abs, worst_rel_l1, skipped_pixels,
     used_values) = compare_columns(ref, got, columns, args.ignore_nonfinite)
    integrated = integrated_comparison(ref, got, args.ignore_nonfinite)

    print(f"reference: {args.reference}")
    print(f"candidate: {args.candidate}")
    print(f"pixels: {len(ref)}")
    if args.ignore_nonfinite:
        print(f"used_values: {used_values}")
        print(f"skipped_nonfinite_pixels: {skipped_pixels}")
    if ref_meta.get("model") or got_meta.get("model"):
        print(f"models: reference={ref_meta.get('model', '')} candidate={got_meta.get('model', '')}")
    for line in lines:
        print(line)
    print(f"worst_Linf={worst_abs:.17e}")
    print(f"worst_rel_L1={worst_rel_l1:.17e}")
    if "nu" in integrated:
        difference = integrated["nu"]["candidate_minus_reference"]
        stokes_difference = difference["stokes_sum_relative"]
        assert isinstance(stokes_difference, dict)
        print("integrated_nu_relative=" + ",".join(
            f"{stokes}:{stokes_difference[stokes]:.17e}" for stokes in "IQUV"))
        print(
            "integrated_nu_polarization="
            f"dLP:{difference['linear_polarization_fraction_absolute']:.17e},"
            f"dCP:{difference['circular_polarization_fraction_absolute']:.17e},"
            f"dEVPA_deg:{difference['evpa_deg_wrapped']:.17e}")

    if args.json_output is not None:
        payload = {
            "schema": "kpolaris_stokes_csv_comparison",
            "schema_version": 1,
            "reference": str(args.reference),
            "candidate": str(args.candidate),
            "reference_metadata": ref_meta,
            "candidate_metadata": got_meta,
            "pixels": len(ref),
            "columns": columns,
            "ignore_nonfinite": args.ignore_nonfinite,
            "used_values": used_values,
            "skipped_nonfinite_pixels": skipped_pixels,
            "metrics": metrics,
            "nmse_definition": ("sum((candidate-reference)^2)/sum(reference^2); "
                                "zero/zero defined as 0"),
            "integrated": integrated,
            "worst_linf": worst_abs,
            "worst_relative_l1": worst_rel_l1,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    if args.atol is not None and worst_abs > args.atol:
        return 2
    if args.rtol_l1 is not None and worst_rel_l1 > args.rtol_l1:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
