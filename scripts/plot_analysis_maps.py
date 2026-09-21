#!/usr/bin/env python3
"""Plot KPolaris physical analysis maps from HDF5 image output."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import LogLocator
import numpy as np
from kpolaris_image import load_image, image_extent as native_extent


LEGACY_DEFAULT_FIELDS = [
    "emission_weight",
    "emission_weighted_radius",
    "emission_weighted_optical_depth_to_camera",
    "absorption_depth",
    "faraday_rotation_depth",
    "faraday_conversion_depth",
    "dominant_emission_radius",
    "dominant_emission_region",
    "emission_weighted_ne_cgs",
    "emission_weighted_thetae",
    "photon_ring_winding_estimate",
    "radiation_substeps",
]

FORMATION_DEFAULT_FIELDS = [
    "observer_weighted_radius_I",
    "intensity_formation_radius_median",
    "observer_weighted_radius_linear",
    "linear_formation_radius_median",
    "observer_weighted_radius_circular",
    "circular_formation_radius_median",
    "los_linear_coherence",
    "los_circular_coherence",
    "foreground_faraday_operator_fraction",
    "absorption_depth",
    "faraday_rotation_depth",
    "faraday_conversion_depth",
]

SIGNED_FIELDS = {
    "faraday_rotation_depth",
    "photon_ring_winding_estimate",
}

LINEAR_FIELDS = {
    "dominant_emission_region",
    "radiation_substeps",
    "observer_weighted_radius_I",
    "intensity_formation_radius_median",
    "observer_weighted_radius_linear",
    "linear_formation_radius_median",
    "observer_weighted_radius_circular",
    "circular_formation_radius_median",
    "los_linear_coherence",
    "los_circular_coherence",
    "foreground_faraday_operator_fraction",
    "contribution_closure_relative_l1",
}

UNIT_INTERVAL_FIELDS = {
    "los_linear_coherence",
    "los_circular_coherence",
    "foreground_faraday_operator_fraction",
}


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def scalar_attr(value, default=np.nan):
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr)
        if arr.size > 0:
            return float(arr.flat[0])
    except (TypeError, ValueError):
        pass
    return default


def read_attr(obj, name, default=np.nan):
    if obj is not None and name in obj.attrs:
        return scalar_attr(obj.attrs[name], default)
    return default


def colorbar(mappable):
    return mappable.axes.figure.colorbar(mappable, ax=mappable.axes, fraction=.046, pad=.04)


def finite_limits(data: np.ndarray, lo: float = 1.0, hi: float = 99.5):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(valid, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def positive_norm(data: np.ndarray):
    valid = data[np.isfinite(data) & (data > 0)]
    if valid.size == 0:
        return None
    vmin, vmax = np.nanpercentile(valid, [1.0, 99.5])
    vmin = max(float(vmin), 1.0e-300)
    vmax = max(float(vmax), vmin * 10.0)
    return LogNorm(vmin=vmin, vmax=vmax)


def signed_norm(data: np.ndarray):
    valid = np.abs(data[np.isfinite(data)])
    valid = valid[valid > 0]
    if valid.size == 0:
        return None
    vmax = max(float(np.nanpercentile(valid, 99.5)), 1.0e-300)
    return SymLogNorm(linthresh=vmax * 1.0e-3, vmin=-vmax, vmax=vmax)


def plot_panel(ax, data: np.ndarray, name: str, extent, interpolation: str):
    if name in SIGNED_FIELDS:
        cmap = "coolwarm"
        norm = signed_norm(data)
        kwargs = {}
    elif name in LINEAR_FIELDS:
        cmap = "viridis"
        norm = None
        if name in UNIT_INTERVAL_FIELDS:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = finite_limits(data)
        kwargs = {"vmin": vmin, "vmax": vmax}
    else:
        cmap = "magma"
        norm = positive_norm(data)
        kwargs = {}
    if norm is not None:
        kwargs["norm"] = norm
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, interpolation=interpolation, **kwargs)
    titles = {
        "observer_weighted_radius_I": "Mean intensity formation radius",
        "intensity_formation_radius_median": "Median intensity formation radius",
        "absorption_depth": "Absorption depth",
        "faraday_rotation_depth": "Faraday rotation depth",
        "faraday_conversion_depth": "Faraday conversion depth",
    }
    ax.set_title(titles.get(name, name.replace("_", " ").capitalize()), fontsize=10, wrap=True)
    ax.set_aspect("equal")
    bar = colorbar(im)
    bar.ax.tick_params(labelsize=9)
    if isinstance(norm, SymLogNorm):
        bar.set_label("symmetric log scale", fontsize=9)
        outer = 10. ** np.floor(np.log10(norm.vmax))
        bar.set_ticks([-outer, -outer / 10, 0, outer / 10, outer])
    elif isinstance(norm, LogNorm):
        bar.set_label("log scale", fontsize=9)
        bar.locator = LogLocator(numticks=5)
        bar.update_ticks()
    elif "radius" in name:
        bar.set_label(r"$GM/c^2$", fontsize=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fields", default=None,
                        help="Comma-separated analysis datasets. Defaults to formation maps for v2 and legacy maps for v1.")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--freq-index", type=int, default=0,
                        help="Frequency group for a multi-frequency analysis file.")
    parser.add_argument("--fov-units", choices=("muas", "M", "pixel"), default="M")
    parser.add_argument("--interpolation", default="nearest")
    args = parser.parse_args()

    image = load_image(args.input, args.frame, args.freq_index)
    with h5py.File(args.input, "r") as h5:
        analysis_path = f"/frame_{args.frame}/freq_{args.freq_index}/analysis"
        if analysis_path not in h5 and args.freq_index == 0:
            analysis_path = f"/frame_{args.frame}/analysis"
        if analysis_path not in h5:
            raise RuntimeError(
                f"input does not contain {analysis_path}; rerun with --analysis_mode=1 "
                "or select an available --freq-index"
            )
        analysis = h5[analysis_path]
        observer_weighted = int(analysis.attrs.get("observer_weighted_available", 0)) == 1
        defaults = FORMATION_DEFAULT_FIELDS if observer_weighted else LEGACY_DEFAULT_FIELDS
        requested = args.fields if args.fields is not None else ",".join(defaults)
        fields = [name.strip() for name in requested.split(",") if name.strip()]
        missing = [name for name in fields if name not in analysis]
        if missing:
            raise RuntimeError("analysis field(s) not found: " + ",".join(missing))
        if not fields:
            raise ValueError("select at least one analysis field")
        maps = []
        for name in fields:
            data = np.asarray(analysis[name], dtype=float)
            if data.shape != image["valid"].shape:
                raise ValueError(f"{name} is not a scalar image map")
            maps.append((name, np.where(image["valid"], data, np.nan)))
        extent, axis_label = native_extent(image, args.fov_units)
        model = decode_attr(h5.attrs["model"]) if "model" in h5.attrs else "unknown"
        frequency = read_attr(analysis, "frequency_hz", np.nan)
        if not np.isfinite(frequency):
            frequency = read_attr(h5.get(f"frame_{args.frame}/freq_{args.freq_index}"), "frequency_hz", np.nan)

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    ncols = min(3, len(maps)) if len(maps) != 4 else 2
    nrows = max(1, int(np.ceil(len(maps) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.4 * nrows), constrained_layout=True,
                             gridspec_kw={"wspace": .25, "hspace": .12})
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, data) in zip(axes, maps):
        plot_panel(ax, data, name, extent, args.interpolation)
        ax.set_xlabel(f"Image x [{axis_label}]")
        ax.set_ylabel(f"Image y [{axis_label}]")
    for ax in axes[len(maps):]:
        ax.axis("off")
    frequency_text = f" · {frequency / 1e9:g} GHz" if np.isfinite(frequency) else ""
    fig.suptitle(
        f"{model.upper()} · Physical diagnostics · frame {args.frame}"
        f" · frequency {args.freq_index}{frequency_text}",
        fontsize=12,
    )
    output = args.output or args.input.with_suffix(".analysis.png")
    if output.resolve() == args.input.resolve():
        raise ValueError("output must differ from input HDF5")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
