#!/usr/bin/env python3
"""Reconstruct KPolaris analysis maps from a full-step trace and compare them."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np


FLOAT_FIELDS = (
    "radiating_path_length",
    "emission_weight",
    "emission_weighted_radius",
    "emission_weighted_optical_depth_to_camera",
    "absorption_depth",
    "absorption_operator_depth",
    "faraday_rotation_depth",
    "faraday_conversion_depth",
    "faraday_operator_depth",
    "dominant_emission_radius",
    "dominant_ne_cgs",
    "dominant_thetae",
    "dominant_b_cgs",
    "dominant_beta",
    "dominant_sigma",
    "emission_weighted_ne_cgs",
    "emission_weighted_thetae",
    "emission_weighted_b_cgs",
    "emission_weighted_beta",
    "emission_weighted_sigma",
    "photon_ring_winding_estimate",
)
INTEGER_FIELDS = ("dominant_emission_region", "radiation_substeps")
ALL_FIELDS = (*FLOAT_FIELDS, *INTEGER_FIELDS)

GEOMETRY_FIELDS = ("dlambda", "r", "phi")
COEFFICIENT_FIELDS = (
    "jI", "aI", "aQ", "aU", "aV", "rhoQ", "rhoU", "rhoV"
)
PLASMA_FIELDS = ("ne_cgs", "thetae", "b_cgs", "beta", "sigma")


def text_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def scalar_attr(group: h5py.Group, name: str) -> float:
    if name not in group.attrs:
        raise ValueError(f"missing required attribute {group.name}:{name}")
    return float(group.attrs[name])


def require_matching_attributes(
    image_group: h5py.Group,
    trace_group: h5py.Group,
    names: tuple[str, ...],
) -> None:
    for name in names:
        if name not in image_group.attrs or name not in trace_group.attrs:
            raise ValueError(
                f"missing required matching attribute {name!r} in "
                f"{image_group.name} or {trace_group.name}"
            )
        image_value = image_group.attrs[name]
        trace_value = trace_group.attrs[name]
        if isinstance(image_value, (bytes, str)) or isinstance(trace_value, (bytes, str)):
            matches = text_value(image_value) == text_value(trace_value)
        else:
            matches = bool(np.array_equal(np.asarray(image_value), np.asarray(trace_value)))
        if not matches:
            raise ValueError(
                f"image and trace parameter {name!r} differ: "
                f"{image_value!r} != {trace_value!r}"
            )


def trace_groups(h5: h5py.File, freq_index: int) -> tuple[h5py.Group, h5py.Group]:
    version = int(h5.attrs["schema_version"])
    nfreq = int(h5.attrs.get("nfreq", 1))
    if version in {1, 2} and nfreq == 1:
        if freq_index != 0:
            raise ValueError("single-frequency trace contains only frequency index 0")
        group = h5["/trace"]
        return group, group
    if version in {2, 3} and nfreq > 1:
        frequency_name = f"/trace/freq_{freq_index}"
        if frequency_name not in h5:
            raise ValueError(f"trace does not contain frequency index {freq_index}")
        return h5["/trace/shared"], h5[frequency_name]
    raise ValueError(f"unsupported kpolaris_trace schema version {version}")


def ray_field(
    dataset: h5py.Dataset,
    ray: int,
    count: int,
    offsets: np.ndarray | None,
) -> np.ndarray:
    if offsets is None:
        if dataset.ndim != 2:
            raise ValueError(f"dense dataset {dataset.name} must be two-dimensional")
        return np.asarray(dataset[ray, :count], dtype=np.float64)
    if dataset.ndim != 1:
        raise ValueError(f"ragged dataset {dataset.name} must be one-dimensional")
    begin, end = int(offsets[ray]), int(offsets[ray + 1])
    if end - begin != count:
        raise ValueError(
            f"{dataset.name}: offset length {end - begin} != sample_count {count} for ray {ray}"
        )
    return np.asarray(dataset[begin:end], dtype=np.float64)


def validate_inputs(
    image: h5py.File, trace: h5py.File, freq_index: int
) -> tuple[int, int, h5py.Group, h5py.Group]:
    if text_value(image.attrs.get("schema", "")) != "kpolaris_image":
        raise ValueError("analysis input is not schema='kpolaris_image'")
    if int(image.attrs.get("schema_version", 0)) != 1:
        raise ValueError("only kpolaris_image schema version 1 is supported")
    if "/frame_0/analysis" not in image:
        raise ValueError("image has no /frame_0/analysis group")
    analysis = image["/frame_0/analysis"]
    if text_value(analysis.attrs.get("schema", "")) != "kpolaris_physical_analysis":
        raise ValueError("unexpected physical-analysis schema")
    if int(analysis.attrs.get("schema_version", 0)) not in {1, 2}:
        raise ValueError("only kpolaris_physical_analysis versions 1 and 2 are supported")

    if text_value(trace.attrs.get("schema", "")) != "kpolaris_trace":
        raise ValueError("trace input is not schema='kpolaris_trace'")
    if int(trace.attrs.get("trace_stride", 0)) != 1:
        raise ValueError("offline reconstruction requires trace_stride=1")
    if text_value(trace.attrs.get("trace_precision", "")) not in {"double", "float64"}:
        raise ValueError("offline golden reconstruction requires trace_precision=double")

    nx = int(image.attrs["nx"])
    ny = int(image.attrs["ny"])
    if int(trace.attrs["nx"]) != nx or int(trace.attrs["ny"]) != ny:
        raise ValueError("image and trace dimensions differ")
    for name in ("model", "camera", "coordinate"):
        if text_value(image.attrs[name]) != text_value(trace.attrs[name]):
            raise ValueError(f"image and trace {name} values differ")

    require_matching_attributes(
        image["/parameters/spacetime"],
        trace["/parameters/spacetime"],
        ("coordinate", "mass", "spin"),
    )
    require_matching_attributes(
        image["/parameters/camera"],
        trace["/parameters/camera"],
        (
            "model", "nx", "ny", "radius", "inclination_rad", "fov", "fovy",
            "x_offset", "y_offset", "use_pinhole_pixel_bias", "pinhole_pixel_bias",
        ),
    )
    require_matching_attributes(
        image["/parameters/integration"],
        trace["/parameters/integration"],
        (
            "step", "adaptive", "adaptive_tolerance", "min_step", "max_step",
            "max_radiation_step", "max_radiation_depth", "max_absorption_depth",
            "max_faraday_depth", "max_steps", "inner_radius", "outer_radius",
        ),
    )
    if text_value(image.attrs["model"]) == "riaf":
        require_matching_attributes(
            image["/parameters/radiation"],
            trace["/parameters/radiation"],
            (
                "model", "emission_type", "emission_fit", "riaf_r_min", "riaf_r_max",
                "riaf_nth0", "riaf_Te0", "riaf_disk_h", "riaf_pow_nth", "riaf_pow_T",
                "riaf_ne_unit", "riaf_te_unit", "riaf_mbh_solar",
                "riaf_keplerian_factor", "riaf_infall_factor", "nonthermal_kappa",
                "variable_kappa", "variable_kappa_min", "variable_kappa_interp_start",
                "variable_kappa_max", "powerlaw_p", "powerlaw_eta",
                "powerlaw_gamma_min", "powerlaw_gamma_max", "powerlaw_gamma_cutoff",
            ),
        )

    geometry, frequency = trace_groups(trace, freq_index)
    image_frequency = scalar_attr(image[f"/frame_0/freq_{freq_index}"], "frequency_hz")
    if int(trace.attrs.get("nfreq", 1)) == 1:
        trace_frequency = float(trace.attrs["frequency_hz"])
    else:
        trace_frequency = scalar_attr(frequency, "frequency_hz")
    if not math.isclose(image_frequency, trace_frequency, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("image and trace observing frequencies differ")

    missing_geometry = [name for name in GEOMETRY_FIELDS if name not in geometry]
    missing_frequency = [
        name for name in (*COEFFICIENT_FIELDS, *PLASMA_FIELDS) if name not in frequency
    ]
    if missing_geometry or missing_frequency:
        raise ValueError(
            "trace is missing reconstruction fields: "
            + ",".join((*missing_geometry, *missing_frequency))
        )
    return nx, ny, geometry, frequency


def reconstruct(
    image: h5py.File,
    trace: h5py.File,
    freq_index: int,
) -> dict[str, np.ndarray]:
    nx, ny, geometry, frequency = validate_inputs(image, trace, freq_index)
    npix = nx * ny
    rays = trace["/rays"]
    pixels = np.asarray(rays["pixel"], dtype=np.int64)
    counts = np.asarray(rays["sample_count"], dtype=np.int64)
    pass_b_steps = np.asarray(rays["pass_b_steps"], dtype=np.int64)
    if pixels.shape != (npix,) or set(pixels.tolist()) != set(range(npix)):
        raise ValueError("trace_mode=image must contain every image pixel exactly once")
    if not np.array_equal(counts, pass_b_steps):
        bad = int(np.flatnonzero(counts != pass_b_steps)[0])
        raise ValueError(
            "trace is truncated or not full-step: "
            f"ray {bad} has {counts[bad]} samples and {pass_b_steps[bad]} Pass B steps"
        )

    layout = text_value(trace.attrs["trace_layout"])
    if layout == "ragged":
        if "sample_offset" not in rays:
            raise ValueError("ragged trace is missing /rays/sample_offset")
        offsets: np.ndarray | None = np.asarray(rays["sample_offset"], dtype=np.int64)
        if offsets.shape != (npix + 1,) or offsets[0] != 0 or offsets[-1] != counts.sum():
            raise ValueError("invalid ragged sample_offset array")
    elif layout == "dense":
        offsets = None
    else:
        raise ValueError(f"unsupported trace layout {layout!r}")

    integration = trace["/parameters/integration"]
    inner_radius = scalar_attr(integration, "inner_radius")
    outer_radius = scalar_attr(integration, "outer_radius")

    output = {
        name: np.zeros(npix, dtype=np.float64) for name in FLOAT_FIELDS
    }
    output.update({name: np.zeros(npix, dtype=np.int32) for name in INTEGER_FIELDS})

    for ray, pixel_value in enumerate(pixels):
        pixel = int(pixel_value)
        count = int(counts[ray])
        values: dict[str, np.ndarray] = {}
        for name in GEOMETRY_FIELDS:
            values[name] = ray_field(geometry[name], ray, count, offsets)
        for name in (*COEFFICIENT_FIELDS, *PLASMA_FIELDS):
            values[name] = ray_field(frequency[name], ray, count, offsets)

        dl = np.abs(values["dlambda"])
        active = (
            (values["r"] >= inner_radius)
            & (values["r"] < outer_radius)
            & (dl > 0.0)
        )
        active_dl = np.where(active, dl, 0.0)
        absorption_delta = np.maximum(values["aI"], 0.0) * active_dl
        absorption_operator_delta = (
            np.abs(values["aI"])
            + np.sqrt(values["aQ"] ** 2 + values["aU"] ** 2 + values["aV"] ** 2)
        ) * active_dl
        faraday_rotation_delta = values["rhoV"] * active_dl
        faraday_conversion_delta = np.sqrt(
            values["rhoQ"] ** 2 + values["rhoU"] ** 2
        ) * active_dl
        faraday_operator_delta = np.sqrt(
            values["rhoQ"] ** 2 + values["rhoU"] ** 2 + values["rhoV"] ** 2
        ) * active_dl
        weight = np.maximum(values["jI"], 0.0) * active_dl

        output["radiating_path_length"][pixel] = active_dl.sum(dtype=np.float64)
        output["emission_weight"][pixel] = weight.sum(dtype=np.float64)
        output["absorption_depth"][pixel] = absorption_delta.sum(dtype=np.float64)
        output["absorption_operator_depth"][pixel] = absorption_operator_delta.sum(dtype=np.float64)
        output["faraday_rotation_depth"][pixel] = faraday_rotation_delta.sum(dtype=np.float64)
        output["faraday_conversion_depth"][pixel] = faraday_conversion_delta.sum(dtype=np.float64)
        output["faraday_operator_depth"][pixel] = faraday_operator_delta.sum(dtype=np.float64)
        output["radiation_substeps"][pixel] = int(np.count_nonzero(active))

        if count > 1:
            delta_phi = np.diff(values["phi"])
            delta_phi = np.where(delta_phi > np.pi, delta_phi - 2.0 * np.pi, delta_phi)
            delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2.0 * np.pi, delta_phi)
            output["photon_ring_winding_estimate"][pixel] = (
                np.abs(delta_phi).sum(dtype=np.float64) / (2.0 * np.pi)
            )

        total_weight = output["emission_weight"][pixel]
        if total_weight <= 0.0:
            continue

        output["emission_weighted_radius"][pixel] = (
            (weight * values["r"]).sum(dtype=np.float64) / total_weight
        )
        absorption_before = np.cumsum(absorption_delta, dtype=np.float64) - absorption_delta
        source_mid_depth = absorption_before + 0.5 * absorption_delta
        mean_source_depth = (weight * source_mid_depth).sum(dtype=np.float64) / total_weight
        output["emission_weighted_optical_depth_to_camera"][pixel] = max(
            0.0, output["absorption_depth"][pixel] - mean_source_depth
        )

        dominant = int(np.argmax(weight))
        if weight[dominant] > 0.0:
            radius = values["r"][dominant]
            output["dominant_emission_radius"][pixel] = radius
            output["dominant_emission_region"][pixel] = (
                1 if radius < 5.0 else (2 if radius < 20.0 else 3)
            )
            plasma_valid = (
                values["ne_cgs"][dominant] > 0.0
                and values["thetae"][dominant] > 0.0
                and values["b_cgs"][dominant] > 0.0
            )
            if plasma_valid:
                for suffix, source in (
                    ("ne_cgs", "ne_cgs"),
                    ("thetae", "thetae"),
                    ("b_cgs", "b_cgs"),
                    ("beta", "beta"),
                    ("sigma", "sigma"),
                ):
                    output[f"dominant_{suffix}"][pixel] = values[source][dominant]

        plasma_valid = (
            (values["ne_cgs"] > 0.0)
            & (values["thetae"] > 0.0)
            & (values["b_cgs"] > 0.0)
        )
        valid_weight = np.where(plasma_valid, weight, 0.0)
        for suffix, source in (
            ("ne_cgs", "ne_cgs"),
            ("thetae", "thetae"),
            ("b_cgs", "b_cgs"),
            ("beta", "beta"),
            ("sigma", "sigma"),
        ):
            output[f"emission_weighted_{suffix}"][pixel] = (
                (valid_weight * values[source]).sum(dtype=np.float64) / total_weight
            )

    for name in output:
        output[name] = output[name].reshape(ny, nx)

    image_reason = np.asarray(image["/frame_0/diagnostics/reason"], dtype=np.int32).reshape(-1)
    image_pass_a = np.asarray(image["/frame_0/diagnostics/pass_a_steps"], dtype=np.int32).reshape(-1)
    image_pass_b = np.asarray(image["/frame_0/diagnostics/steps"], dtype=np.int32).reshape(-1)
    trace_reason = np.asarray(rays["reason"], dtype=np.int32)
    trace_pass_a = np.asarray(rays["pass_a_steps"], dtype=np.int32)
    for ray, pixel_value in enumerate(pixels):
        pixel = int(pixel_value)
        if (
            image_reason[pixel] != trace_reason[ray]
            or image_pass_a[pixel] != trace_pass_a[ray]
            or image_pass_b[pixel] != pass_b_steps[ray]
        ):
            raise ValueError(f"image/trace discrete diagnostics differ at pixel {pixel}")
    return output


def compare(
    analysis: h5py.Group,
    reconstructed: dict[str, np.ndarray],
    rtol: float,
    atol: float,
) -> tuple[dict[str, dict[str, float | int | bool]], list[str]]:
    metrics: dict[str, dict[str, float | int | bool]] = {}
    failed: list[str] = []
    for name in ALL_FIELDS:
        expected = np.asarray(analysis[name])
        actual = reconstructed[name]
        if name in INTEGER_FIELDS:
            mismatch = expected != actual
            count = int(np.count_nonzero(mismatch))
            metrics[name] = {
                "exact": count == 0,
                "mismatched_pixels": count,
                "max_absolute_difference": float(
                    np.max(np.abs(expected.astype(np.int64) - actual.astype(np.int64)), initial=0)
                ),
            }
            if count:
                failed.append(name)
            continue

        difference = actual - expected
        absolute = np.abs(difference)
        denominator = float(np.abs(expected).sum(dtype=np.float64))
        reference_peak = float(np.max(np.abs(expected), initial=0.0))
        passed = bool(np.all(np.isfinite(actual)) and np.allclose(actual, expected, rtol=rtol, atol=atol))
        metrics[name] = {
            "passed": passed,
            "max_absolute_difference": float(np.max(absolute, initial=0.0)),
            "l1_relative": float(absolute.sum(dtype=np.float64) / denominator) if denominator else 0.0,
            "linf_over_reference_peak": float(np.max(absolute, initial=0.0) / reference_peak)
            if reference_peak else 0.0,
        }
        if not passed:
            failed.append(name)
    return metrics, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_image", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--freq-index", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-300)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.freq_index < 0:
        parser.error("--freq-index must be non-negative")
    if args.rtol < 0.0 or args.atol < 0.0:
        parser.error("--rtol and --atol must be non-negative")

    try:
        with h5py.File(args.analysis_image, "r") as image, h5py.File(args.trace, "r") as trace:
            reconstructed = reconstruct(image, trace, args.freq_index)
            metrics, failed = compare(image["/frame_0/analysis"], reconstructed, args.rtol, args.atol)
            payload = {
                "schema": "kpolaris_analysis_trace_comparison",
                "schema_version": 1,
                "analysis_image": str(args.analysis_image),
                "trace": str(args.trace),
                "frequency_index": args.freq_index,
                "ray_count": int(trace.attrs["ray_count"]),
                "rtol": args.rtol,
                "atol": args.atol,
                "fields": metrics,
                "failed_fields": failed,
                "passed": not failed,
            }
    except (OSError, KeyError, ValueError) as error:
        print(f"analysis/trace comparison error: {error}", file=sys.stderr)
        return 2

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("field l1_relative linf_over_reference_peak max_absolute_difference status")
    for name in ALL_FIELDS:
        row = metrics[name]
        if name in INTEGER_FIELDS:
            status = "pass" if row["exact"] else "FAIL"
            print(f"{name} exact exact {row['max_absolute_difference']:.17e} {status}")
        else:
            status = "pass" if row["passed"] else "FAIL"
            print(
                f"{name} {row['l1_relative']:.17e} "
                f"{row['linf_over_reference_peak']:.17e} "
                f"{row['max_absolute_difference']:.17e} {status}"
            )
    if failed:
        print("failed_fields=" + ",".join(failed), file=sys.stderr)
        return 1
    print("analysis_trace_comparison=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
