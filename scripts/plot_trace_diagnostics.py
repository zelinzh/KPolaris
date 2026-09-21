#!/usr/bin/env python3
"""Plot KPolaris trace diagnostics from dense or ragged HDF5 trace files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def attr(h5, name, default=None):
    return decode_attr(h5.attrs[name]) if name in h5.attrs else default


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return ax.figure.colorbar(mappable, cax=cax)


def ray_image(h5, name: str, nx: int, ny: int) -> np.ndarray:
    values = np.asarray(h5[f"/rays/{name}"])
    ix = np.asarray(h5["/rays/ix"], dtype=np.int64)
    iy = np.asarray(h5["/rays/iy"], dtype=np.int64)
    image = np.full((ny, nx), np.nan, dtype=float)
    image[iy, ix] = values.astype(float)
    return image


def finite_limits(data: np.ndarray, lo: float = 1.0, hi: float = 99.5):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(valid, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def positive_log_norm(data: np.ndarray):
    valid = data[np.isfinite(data) & (data > 0)]
    if valid.size == 0:
        return None
    vmin, vmax = np.nanpercentile(valid, [1, 99.5])
    vmin = max(float(vmin), 1.0e-30)
    vmax = max(float(vmax), vmin * 10.0)
    return LogNorm(vmin=vmin, vmax=vmax)


def trace_dataset(h5, field: str, freq_index: int = 0):
    if field.startswith("derived/"):
        name = field.split("/", 1)[1]
        group = derived_group(h5, freq_index)
        return group.get(name) if group is not None else None
    paths = [f"/trace/freq_{freq_index}/{field}", f"/trace/shared/{field}"]
    if freq_index == 0:
        paths.append(f"/trace/{field}")
    for path in paths:
        if path in h5:
            return h5[path]
    return None


def derived_group(h5, freq_index: int = 0):
    paths = [f"/trace/freq_{freq_index}/derived"]
    if freq_index == 0:
        paths.append("/derived")
    for path in paths:
        if path in h5:
            return h5[path]
    return None


def ray_samples(h5, dataset, ray_index: int) -> np.ndarray:
    counts = np.asarray(h5["/rays/sample_count"], dtype=np.int64)
    count = int(counts[ray_index])
    if dataset.ndim == 2:
        return np.asarray(dataset[ray_index, :min(count, dataset.shape[1])], dtype=float)
    if dataset.ndim == 1:
        offsets = np.asarray(h5["/rays/sample_offset"], dtype=np.int64)
        return np.asarray(dataset[int(offsets[ray_index]):int(offsets[ray_index + 1])], dtype=float)
    raise ValueError(f"unsupported history dataset rank: {dataset.ndim}")


def select_history_ray(h5, ray_index, ix, iy) -> int:
    nray = int(np.asarray(h5["/rays/ix"]).size)
    if (ix is None) != (iy is None):
        raise RuntimeError("--history-ix and --history-iy must be specified together")
    if ix is not None:
        ray_ix = np.asarray(h5["/rays/ix"], dtype=np.int64)
        ray_iy = np.asarray(h5["/rays/iy"], dtype=np.int64)
        matches = np.flatnonzero((ray_ix == ix) & (ray_iy == iy))
        if matches.size != 1:
            raise RuntimeError(f"trace contains {matches.size} rays at pixel ({ix},{iy})")
        selected = int(matches[0])
        if ray_index is not None and selected != ray_index:
            raise RuntimeError("--history-ray-index and --history-ix/iy select different rays")
        return selected
    selected = 0 if ray_index is None else ray_index
    if selected < 0 or selected >= nray:
        raise RuntimeError(f"--history-ray-index {selected} is outside [0,{nray})")
    return selected


def read_history(h5, freq_index: int, ray_index: int):
    derived = derived_group(h5, freq_index)
    if derived is None or int(derived.attrs.get("available", 0)) != 1:
        raise RuntimeError("trace has no available derived polarization histories")
    names = [
        "intensity_fraction_of_final",
        "linear_fraction",
        "circular_fraction",
        "evpa_unwrapped_rad",
        "cumulative_absorption_depth",
        "cumulative_faraday_rotation_depth",
        "cumulative_faraday_conversion_depth",
        "cumulative_faraday_operator_depth",
    ]
    missing = [name for name in names if name not in derived]
    lam = trace_dataset(h5, "lambda", freq_index)
    if lam is None:
        missing.append("lambda")
    if missing:
        raise RuntimeError("trace history field(s) not found: " + ",".join(missing))
    values = {name: ray_samples(h5, derived[name], ray_index) for name in names}
    values["lambda"] = ray_samples(h5, lam, ray_index)
    lengths = {name: data.size for name, data in values.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"trace history arrays have inconsistent lengths: {lengths}")
    values["freeze_lambda"] = float(np.asarray(derived["intensity_freeze_lambda"])[ray_index])
    values["complete"] = int(np.asarray(derived["complete"])[ray_index])
    values["ix"] = int(np.asarray(h5["/rays/ix"])[ray_index])
    values["iy"] = int(np.asarray(h5["/rays/iy"])[ray_index])
    values["ray_index"] = ray_index
    return values


def plot_history(history, output: Path, frequency):
    lam = history["lambda"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    panels = axes.flat
    panels[0].plot(lam, history["intensity_fraction_of_final"], color="black")
    panels[0].axhline(1.0, color="0.6", linestyle="--", linewidth=0.8)
    panels[0].set_ylabel("I / I_final")
    panels[1].plot(lam, history["linear_fraction"], label="m_L")
    panels[1].plot(lam, history["circular_fraction"], label="m_C")
    panels[1].legend(fontsize=8)
    panels[1].set_ylabel("polarization fraction")
    panels[2].plot(lam, history["evpa_unwrapped_rad"], color="tab:purple")
    panels[2].set_ylabel("unwrapped EVPA [rad]")
    panels[3].plot(lam, history["cumulative_absorption_depth"], color="tab:orange")
    panels[3].set_ylabel("cumulative absorption depth")
    panels[4].plot(lam, history["cumulative_faraday_rotation_depth"], color="tab:red")
    panels[4].set_ylabel("signed cumulative rotation depth")
    panels[5].plot(lam, history["cumulative_faraday_conversion_depth"], label="conversion")
    panels[5].plot(lam, history["cumulative_faraday_operator_depth"], label="operator")
    panels[5].legend(fontsize=8)
    panels[5].set_ylabel("cumulative Faraday depth")
    freeze = history["freeze_lambda"]
    for ax in panels:
        ax.set_xlabel("lambda")
        ax.grid(alpha=0.2)
        if np.isfinite(freeze):
            ax.axvline(freeze, color="tab:green", linestyle=":", linewidth=1.0)
    frequency_text = f", nu={frequency:.6g} Hz" if frequency is not None else ""
    fig.suptitle(
        f"Polarization history: ray={history['ray_index']} pixel=({history['ix']},{history['iy']}), "
        f"complete={history['complete']}{frequency_text}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def trace_field_stat(h5, field: str, stat: str, freq_index: int = 0) -> np.ndarray:
    data = trace_dataset(h5, field, freq_index)
    if data is None:
        raise KeyError(f"trace field not found: {field}")
    counts = np.asarray(h5["/rays/sample_count"], dtype=np.int64)
    out = np.full(counts.shape, np.nan, dtype=float)
    if data.ndim == 2:
        max_samples = data.shape[1]
        for i, count in enumerate(counts):
            n = min(int(count), max_samples)
            if n <= 0:
                continue
            values = np.asarray(data[i, :n], dtype=float)
            out[i] = reduce_values(values, stat)
        return out
    if data.ndim == 1:
        if "/rays/sample_offset" not in h5:
            raise KeyError("ragged trace field requires /rays/sample_offset")
        offsets = np.asarray(h5["/rays/sample_offset"], dtype=np.int64)
        for i in range(counts.size):
            start = int(offsets[i])
            stop = int(offsets[i + 1])
            if stop <= start:
                continue
            values = np.asarray(data[start:stop], dtype=float)
            out[i] = reduce_values(values, stat)
        return out
    raise ValueError(f"unsupported trace field rank for {field}: {data.ndim}")


def reduce_values(values: np.ndarray, stat: str) -> float:
    if stat == "mean":
        return float(np.nanmean(values))
    if stat == "min":
        return float(np.nanmin(values))
    if stat == "max":
        return float(np.nanmax(values))
    if stat == "last":
        return float(values[-1])
    if stat == "first":
        return float(values[0])
    raise ValueError(f"unknown field statistic: {stat}")


def field_stat_image(h5, field: str, stat: str, nx: int, ny: int, freq_index: int = 0) -> np.ndarray:
    values = trace_field_stat(h5, field, stat, freq_index)
    ix = np.asarray(h5["/rays/ix"], dtype=np.int64)
    iy = np.asarray(h5["/rays/iy"], dtype=np.int64)
    image = np.full((ny, nx), np.nan, dtype=float)
    image[iy, ix] = values
    return image


def plot_panel(ax, data: np.ndarray, title: str, cmap: str = "viridis", log: bool = False):
    norm = positive_log_norm(data) if log else None
    kwargs = {"origin": "lower", "cmap": cmap, "interpolation": "nearest"}
    if norm is not None:
        kwargs["norm"] = norm
    else:
        vmin, vmax = finite_limits(data)
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    im = ax.imshow(data, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar(im).ax.tick_params(labelsize=7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--field", default=None,
                        help="Optional trace field to summarize per ray, e.g. r, jI, rhoV.")
    parser.add_argument("--field-stat", choices=("mean", "min", "max", "first", "last"), default="mean")
    parser.add_argument("--log-field", action="store_true")
    parser.add_argument("--freq-index", type=int, default=0,
                        help="Frequency group to use for frequency-dependent fields in multi-frequency trace files.")
    parser.add_argument("--history-ray-index", type=int,
                        help="Plot the derived path history for this row of /rays instead of summary maps.")
    parser.add_argument("--history-ix", type=int,
                        help="Select a history ray by image x pixel; requires --history-iy.")
    parser.add_argument("--history-iy", type=int,
                        help="Select a history ray by image y pixel; requires --history-ix.")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as h5:
        nx = int(attr(h5, "nx"))
        ny = int(attr(h5, "ny"))
        layout = attr(h5, "trace_layout", "dense")
        compression = int(attr(h5, "trace_compression", 0))
        precision = attr(h5, "trace_precision", "unknown")
        fields = attr(h5, "trace_fields", "")
        nfreq = int(attr(h5, "nfreq", 1))
        if args.freq_index < 0 or args.freq_index >= nfreq:
            raise RuntimeError(f"--freq-index {args.freq_index} is outside trace nfreq={nfreq}")
        frequency = None
        freq_path = f"/trace/freq_{args.freq_index}"
        if freq_path in h5 and "frequency_hz" in h5[freq_path].attrs:
            frequency = float(h5[freq_path].attrs["frequency_hz"])
        elif "/grid/frequency_hz" in h5:
            freqs = np.asarray(h5["/grid/frequency_hz"], dtype=float)
            if args.freq_index < freqs.size:
                frequency = float(freqs[args.freq_index])
        elif "frequency_hz" in h5.attrs:
            frequency = float(h5.attrs["frequency_hz"])

        history_requested = (
            args.history_ray_index is not None or
            args.history_ix is not None or args.history_iy is not None
        )
        history = None
        if history_requested:
            selected = select_history_ray(
                h5, args.history_ray_index, args.history_ix, args.history_iy
            )
            history = read_history(h5, args.freq_index, selected)

        panels = [
            (ray_image(h5, "sample_count", nx, ny), "sample_count", "viridis", False),
            (ray_image(h5, "pass_a_steps", nx, ny), "pass_a_steps", "cividis", False),
            (ray_image(h5, "pass_b_steps", nx, ny), "pass_b_steps", "magma", False),
            (ray_image(h5, "closure_x", nx, ny), "closure_x", "plasma", True),
            (ray_image(h5, "frame_error", nx, ny), "frame_error", "inferno", True),
        ]
        if args.field:
            title = f"{args.field} {args.field_stat}"
            if nfreq > 1:
                title += f" f{args.freq_index}"
            panels.append((field_stat_image(h5, args.field, args.field_stat, nx, ny, args.freq_index),
                           title, "cubehelix", args.log_field))

    output = args.output or args.input.with_suffix(".trace.png")
    if history is not None:
        plot_history(history, output, frequency)
        print(f"wrote {output}")
        return 0

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8), constrained_layout=True)
    for ax, panel in zip(axes.flat, panels):
        plot_panel(ax, *panel)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        f"KPolaris ray diagnostics · frequency index {args.freq_index}" + (f", nu={frequency:.6g} Hz" if frequency is not None else ""),
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
