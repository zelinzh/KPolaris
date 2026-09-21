#!/usr/bin/env python3
"""Bin KPolaris trace samples in black-hole coordinates and render physical maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable
import numpy as np


MEAN_FIELDS = {"ne_cgs", "thetae", "b_cgs", "nu_fluid_hz", "theta_bk", "b1", "b2"}
SIGNED_INTEGRAL_FIELDS = {"rhoQ", "rhoU", "rhoV"}
DELTA_FIELDS = {"dI", "dQ", "dU", "dV"}
DELTA_SOURCE_FIELDS = {"dI": "SI", "dQ": "SQ", "dU": "SU", "dV": "SV"}
FIELD_ALIASES = {
    "ne_mean": ("ne_cgs", "mean"),
    "ne_column": ("ne_cgs", "column"),
    "b_mean": ("b_cgs", "mean"),
    "di": ("dI", "delta"),
    "dq": ("dQ", "delta"),
    "du": ("dU", "delta"),
    "dv": ("dV", "delta"),
    "delta_i": ("dI", "delta"),
    "delta_q": ("dQ", "delta"),
    "delta_u": ("dU", "delta"),
    "delta_v": ("dV", "delta"),
    "deltai": ("dI", "delta"),
    "deltaq": ("dQ", "delta"),
    "deltau": ("dU", "delta"),
    "deltav": ("dV", "delta"),
}
DEFAULT_PANELS = ["dI", "jI", "ne_mean", "ne_cgs_column", "aI", "rhoV", "thetae_mean", "b_cgs_mean", "path_length"]


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def attr(h5, name, default=None):
    return decode_attr(h5.attrs[name]) if name in h5.attrs else default


def parse_number(text: str) -> float:
    value = text.strip().lower()
    if value in {"pi", "+pi"}:
        return float(np.pi)
    if value == "-pi":
        return float(-np.pi)
    if value.startswith("pi/"):
        return float(np.pi) / float(value[3:])
    if value.startswith("-pi/"):
        return -float(np.pi) / float(value[4:])
    if value.endswith("pi"):
        coeff = value[:-2].rstrip("*")
        if coeff in {"", "+"}:
            return float(np.pi)
        if coeff == "-":
            return float(-np.pi)
        return float(coeff) * float(np.pi)
    return float(value)


def parse_range(text: str | None, fallback: tuple[float, float]) -> tuple[float, float]:
    if text is None or text == "":
        return fallback
    parts = [p.strip() for p in text.replace(":", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"range must have two comma-separated values: {text}")
    lo, hi = parse_number(parts[0]), parse_number(parts[1])
    if not hi > lo:
        raise ValueError(f"range upper bound must be greater than lower bound: {text}")
    return lo, hi


def range_mask(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return (values >= lo) & (values <= hi)


def trace_dataset(h5, name: str, freq_index: int = 0):
    for path in (
        f"/trace/{name}",
        f"/trace/shared/{name}",
        f"/trace/freq_{freq_index}/{name}",
    ):
        if path in h5:
            return h5[path]
    return None


def clipped_counts(h5, max_samples: int | None = None) -> np.ndarray:
    counts = np.asarray(h5["/rays/sample_count"], dtype=np.int64)
    if max_samples is None:
        r_data = trace_dataset(h5, "r")
        if r_data is not None and r_data.ndim == 2:
            max_samples = r_data.shape[1]
        else:
            max_samples = int(counts.max()) if counts.size else 0
    return np.minimum(counts, int(max_samples))


def dense_to_flat(dataset, counts: np.ndarray) -> np.ndarray:
    total = int(counts.sum())
    out = np.empty(total, dtype=dataset.dtype)
    offset = 0
    for ray, count in enumerate(counts):
        n = int(count)
        if n <= 0:
            continue
        out[offset:offset + n] = dataset[ray, :n]
        offset += n
    return out


def trace_field_flat(h5, name: str, counts: np.ndarray | None = None, freq_index: int = 0) -> np.ndarray | None:
    data = trace_dataset(h5, name, freq_index)
    if data is None:
        return None
    if data.ndim == 1:
        return np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"unsupported trace field rank for {name}: {data.ndim}")
    if counts is None:
        counts = clipped_counts(h5, data.shape[1])
    return dense_to_flat(data, counts)


def trace_delta_field_flat(h5, name: str, counts: np.ndarray, freq_index: int = 0) -> np.ndarray | None:
    source = DELTA_SOURCE_FIELDS.get(name)
    if source is None:
        return None
    data = trace_dataset(h5, source, freq_index)
    if data is None:
        return None
    total = int(np.sum(counts))
    out = np.full(total, np.nan, dtype=data.dtype)
    dst = 0
    if data.ndim == 1:
        if "/rays/sample_offset" not in h5:
            raise RuntimeError(f"ragged trace delta {name} requires /rays/sample_offset")
        offsets = np.asarray(h5["/rays/sample_offset"], dtype=np.int64)
        for ray, count in enumerate(counts):
            n = int(count)
            if n > 1:
                src = int(offsets[ray])
                out[dst:dst + n - 1] = data[src + 1:src + n] - data[src:src + n - 1]
            dst += n
        return out
    if data.ndim != 2:
        raise ValueError(f"unsupported trace field rank for {source}: {data.ndim}")
    for ray, count in enumerate(counts):
        n = int(count)
        if n > 1:
            out[dst:dst + n - 1] = data[ray, 1:n] - data[ray, :n - 1]
        dst += n
    return out


def midpoint_field_flat(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    offset = 0
    for count in counts:
        n = int(count)
        if n > 1:
            segment = values[offset:offset + n].astype(float)
            out[offset:offset + n - 1] = 0.5 * (segment[:-1] + segment[1:])
        offset += n
    return out


def midpoint_angle_flat(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    twopi = 2.0 * np.pi
    offset = 0
    for count in counts:
        n = int(count)
        if n > 1:
            segment = values[offset:offset + n].astype(float)
            delta = (segment[1:] - segment[:-1] + np.pi) % twopi - np.pi
            out[offset:offset + n - 1] = (segment[:-1] + 0.5 * delta) % twopi
        offset += n
    return out


def is_delta_panel(panel: str) -> bool:
    return field_base(panel)[0] in DELTA_FIELDS


def cartesian_from_spherical(r, theta, phi):
    # These are black-hole centered Cartesian coordinates reconstructed from
    # the BL-like r,theta,phi trace diagnostics, not camera image coordinates.
    sin_th = np.sin(theta)
    x = r * sin_th * np.cos(phi)
    y = r * sin_th * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def finite_mask(*arrays):
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def positive_norm(data):
    valid = data[np.isfinite(data) & (data > 0)]
    if valid.size == 0:
        return None
    lo, hi = np.nanpercentile(valid, [1, 99.7])
    lo = max(float(lo), 1.0e-300)
    hi = max(float(hi), lo * 10.0)
    return LogNorm(vmin=lo, vmax=hi)


def signed_norm(data):
    valid = np.abs(data[np.isfinite(data)])
    valid = valid[valid > 0]
    if valid.size == 0:
        return None
    hi = float(np.nanpercentile(valid, 99.7))
    hi = max(hi, 1.0e-300)
    return SymLogNorm(linthresh=hi * 1.0e-3, vmin=-hi, vmax=hi)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth_histogram(data: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return data
    kernel = gaussian_kernel1d(sigma)
    out = np.asarray(data, dtype=float)
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, out)
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, out)
    return out


def smooth_volume(data: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return data
    kernel = gaussian_kernel1d(sigma)
    out = np.asarray(data, dtype=float)
    for axis in range(3):
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis, out)
    return out


def field_base(field: str) -> tuple[str, str]:
    key = field.strip()
    alias = FIELD_ALIASES.get(key.lower())
    if alias is not None:
        return alias
    if key in DELTA_FIELDS:
        return key, "delta"
    if key == "path_length":
        return key, "path"
    if key.endswith("_column"):
        return key[:-7], "column"
    if key.endswith("_mean"):
        return key[:-5], "mean"
    if key.endswith("_int"):
        return key[:-4], "integral"
    return key, "auto"


def field_values(field: str, fields: dict[str, np.ndarray], weight: np.ndarray):
    base, mode = field_base(field)
    if base not in fields and base != "path_length":
        return None
    if base == "path_length":
        return np.ones_like(weight), "int dl", "inferno", "positive", "integral_positive"
    values = fields[base]
    if base in DELTA_FIELDS or mode == "delta":
        return values, f"sum Delta {base[-1]}", "coolwarm", "signed", "sum"
    if mode == "column":
        return values, f"int {base} dl", "magma", "positive", "integral"
    if mode == "mean":
        return values, f"<{base}>_dl", "viridis", "positive", "mean"
    if mode == "integral":
        signed = base in SIGNED_INTEGRAL_FIELDS
        return values, f"int {base} dl" if signed else f"int max({base},0) dl", "coolwarm" if signed else "magma", "signed" if signed else "positive", "integral_signed" if signed else "integral_positive"
    if base in MEAN_FIELDS:
        return values, f"<{base}>_dl", "viridis", "positive", "mean"
    signed = base in SIGNED_INTEGRAL_FIELDS
    return values, f"int {base} dl" if signed else f"int max({base},0) dl", "coolwarm" if signed else "magma", "signed" if signed else "positive", "integral_signed" if signed else "integral_positive"


def hist2d(a, b, values, weight, range_a, range_b, bins, mode, smooth_sigma=0.0):
    mask = finite_mask(a, b, values, weight)
    mask &= range_mask(a, range_a) & range_mask(b, range_b)
    if mode == "mean":
        num, xedges, yedges = np.histogram2d(a[mask], b[mask], bins=bins, range=[range_a, range_b], weights=values[mask] * weight[mask])
        den, _, _ = np.histogram2d(a[mask], b[mask], bins=bins, range=[range_a, range_b], weights=weight[mask])
        if smooth_sigma > 0:
            num = smooth_histogram(num, smooth_sigma)
            den = smooth_histogram(den, smooth_sigma)
        out = np.full_like(num, np.nan, dtype=float)
        np.divide(num, den, out=out, where=den > 0)
        return out.T, xedges, yedges
    scalar = values[mask]
    if mode == "integral_positive":
        scalar = np.maximum(scalar, 0.0)
    hist_weights = scalar if mode == "sum" else scalar * weight[mask]
    hist, xedges, yedges = np.histogram2d(a[mask], b[mask], bins=bins, range=[range_a, range_b], weights=hist_weights)
    if smooth_sigma > 0:
        hist = smooth_histogram(hist, smooth_sigma)
    return hist.T, xedges, yedges


def make_map_with_edges(field, a, b, fields, weight, range_a, range_b, bins, smooth_sigma=0.0):
    info = field_values(field, fields, weight)
    if info is None:
        return None
    values, title, cmap, scale, mode = info
    data, xedges, yedges = hist2d(a, b, values, weight, range_a, range_b, bins, mode, smooth_sigma)
    return data, title, cmap, scale, xedges, yedges


def make_map(field, a, b, fields, weight, range_a, range_b, bins, smooth_sigma=0.0):
    result = make_map_with_edges(field, a, b, fields, weight, range_a, range_b, bins, smooth_sigma)
    if result is None:
        return None
    data, title, cmap, scale, _xedges, _yedges = result
    return data, title, cmap, scale


def plot_projection(output, maps, range_a, range_b, title, xlabel, ylabel):
    n = len(maps)
    ncols = 3
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (data, panel_title, cmap, scale) in zip(axes, maps):
        if scale == "signed":
            norm = signed_norm(data)
        elif scale == "linear":
            norm = None
        else:
            norm = positive_norm(data)
        kwargs = {"origin": "lower", "extent": [range_a[0], range_a[1], range_b[0], range_b[1]], "cmap": cmap, "interpolation": "nearest", "aspect": "auto"}
        if norm is not None:
            kwargs["norm"] = norm
        else:
            valid = data[np.isfinite(data)]
            if valid.size:
                lo, hi = np.nanpercentile(valid, [1, 99.7])
                if hi <= lo:
                    hi = lo + 1.0
                kwargs["vmin"] = lo
                kwargs["vmax"] = hi
        im = ax.imshow(data, **kwargs)
        ax.axhline(0.0, color="white", lw=0.4, alpha=0.35)
        ax.axvline(0.0, color="white", lw=0.4, alpha=0.35)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    for ax in axes[len(maps):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rtheta_projection(output, maps, r_range, theta_range, title, interpolation="nearest"):
    n = len(maps)
    ncols = 3
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    theta_ticks = [(0.0, "0"), (0.5 * np.pi, "pi/2"), (np.pi, "pi")]
    visible_ticks = [(v, label) for v, label in theta_ticks if theta_range[0] <= v <= theta_range[1]]
    for ax, (data, panel_title, cmap, scale, _redges, _tedges) in zip(axes, maps):
        norm = signed_norm(data) if scale == "signed" else positive_norm(data)
        kwargs = {
            "origin": "lower",
            "extent": [r_range[0], r_range[1], theta_range[0], theta_range[1]],
            "cmap": cmap,
            "interpolation": interpolation,
            "aspect": "auto",
        }
        if norm is not None:
            kwargs["norm"] = norm
        im = ax.imshow(data, **kwargs)
        if theta_range[0] <= 0.5 * np.pi <= theta_range[1]:
            ax.axhline(0.5 * np.pi, color="white", lw=0.5, alpha=0.45)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("r [M]")
        ax.set_ylabel("theta [rad]")
        if visible_ticks:
            ax.set_yticks([v for v, _label in visible_ticks])
            ax.set_yticklabels([label for _v, label in visible_ticks])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    for ax in axes[len(maps):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_meridional_projection(output, maps, title):
    n = len(maps)
    ncols = 3
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (data, panel_title, cmap, scale, redges, tedges) in zip(axes, maps):
        norm = signed_norm(data) if scale == "signed" else positive_norm(data)
        rr, tt = np.meshgrid(redges, tedges)
        cyl_r = rr * np.sin(tt)
        zz = rr * np.cos(tt)
        kwargs = {"cmap": cmap, "shading": "auto"}
        if norm is not None:
            kwargs["norm"] = norm
        im = ax.pcolormesh(cyl_r, zz, data, **kwargs)
        ax.pcolormesh(-cyl_r, zz, data, **kwargs)
        lim = max(float(np.nanmax(np.abs(cyl_r))), float(np.nanmax(np.abs(zz))), 1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0.0, color="white", lw=0.4, alpha=0.35)
        ax.axvline(0.0, color="white", lw=0.4, alpha=0.35)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("R = r sin(theta) [M]")
        ax.set_ylabel("z = r cos(theta) [M]")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    for ax in axes[len(maps):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def hist3d_scalar(x, y, z, scalar_name, fields, weight, ranges, bins):
    x_range, y_range, z_range = ranges
    info = field_values(scalar_name, fields, weight)
    if info is None:
        info = field_values("path_length", fields, weight)
    values, title, _cmap, scale, mode = info
    mask = finite_mask(x, y, z, values, weight)
    mask &= range_mask(x, x_range) & range_mask(y, y_range) & range_mask(z, z_range)
    coords = np.column_stack([x[mask], y[mask], z[mask]])
    if mode == "mean":
        num, edges = np.histogramdd(coords, bins=bins, range=[x_range, y_range, z_range], weights=values[mask] * weight[mask])
        den, _ = np.histogramdd(coords, bins=bins, range=[x_range, y_range, z_range], weights=weight[mask])
        out = np.full_like(num, np.nan, dtype=float)
        np.divide(num, den, out=out, where=den > 0)
        return out, edges, title, scale
    scalar = values[mask]
    if mode == "integral_positive":
        scalar = np.maximum(scalar, 0.0)
    hist_weights = scalar if mode == "sum" else scalar * weight[mask]
    hist, edges = np.histogramdd(coords, bins=bins, range=[x_range, y_range, z_range], weights=hist_weights)
    return hist, edges, title, scale


def log_scaled_values(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.log10(np.maximum(values, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0.0, 1.0)


def volume_color_limits(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        return 1.0, 10.0
    lo = max(float(np.nanpercentile(valid, 2)), 1.0e-300)
    hi = max(float(np.nanpercentile(valid, 99.5)), lo * 10.0)
    return lo, hi


def plot_empty_volume(output, title):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title + " (no nonzero bins)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_3d_volume_voxels(output, hist, edges, title, max_voxels, percentile, cmap_name, display_sigma):
    finite = np.isfinite(hist)
    mag = smooth_volume(np.abs(hist), display_sigma)
    valid_values = mag[finite & (mag > 0)]
    if valid_values.size == 0:
        plot_empty_volume(output, title)
        return
    threshold = np.nanpercentile(valid_values, percentile)
    filled = finite & (mag >= threshold) & (mag > 0)
    if np.count_nonzero(filled) > max_voxels:
        threshold = np.partition(valid_values, -max_voxels)[-max_voxels]
        filled = finite & (mag >= threshold) & (mag > 0)
    selected = mag[filled]
    lo, hi = volume_color_limits(selected)
    normed = log_scaled_values(mag, lo, hi)
    cmap = plt.get_cmap(cmap_name)
    facecolors = cmap(normed)
    facecolors[..., 3] = np.where(filled, 0.08 + 0.72 * normed, 0.0)
    X, Y, Z = np.meshgrid(edges[0], edges[1], edges[2], indexing="ij")
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(X, Y, Z, filled, facecolors=facecolors, edgecolor=None, shade=False)
    ax.set_xlabel("x [M]")
    ax.set_ylabel("y [M]")
    ax.set_zlabel("z [M]")
    ax.set_xlim(edges[0][0], edges[0][-1])
    ax.set_ylim(edges[1][0], edges[1][-1])
    ax.set_zlim(edges[2][0], edges[2][-1])
    ax.view_init(elev=24, azim=-58)
    ax.set_title(f"{title}\nvoxel threshold >= p{percentile:g}, shown={np.count_nonzero(filled)}")
    mappable = ScalarMappable(norm=LogNorm(vmin=lo, vmax=hi), cmap=cmap)
    mappable.set_array([])
    cb = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
    cb.set_label("voxel value")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def centers_from_edges(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def plot_3d_volume_slices(
    output,
    hist,
    edges,
    title,
    max_slices,
    percentile,
    cmap_name,
    display_sigma,
    alpha_power,
    max_alpha,
    projection_walls,
):
    mag = smooth_volume(np.abs(hist), display_sigma)
    valid_values = mag[np.isfinite(mag) & (mag > 0)]
    if valid_values.size == 0:
        plot_empty_volume(output, title)
        return
    threshold = np.nanpercentile(valid_values, percentile)
    lo, hi = volume_color_limits(valid_values[valid_values >= threshold])
    normed = log_scaled_values(mag, lo, hi)
    cmap = plt.get_cmap(cmap_name)

    xc = centers_from_edges(edges[0])
    yc = centers_from_edges(edges[1])
    zc = centers_from_edges(edges[2])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    slice_strength = np.nanmax(mag, axis=(0, 1))
    slice_ids = np.flatnonzero(slice_strength >= threshold)
    if slice_ids.size > max_slices:
        quantiles = np.linspace(0, slice_ids.size - 1, max_slices).round().astype(int)
        slice_ids = slice_ids[quantiles]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for k in slice_ids:
        values = mag[:, :, k]
        alpha = np.where(values >= threshold, max_alpha * np.power(normed[:, :, k], alpha_power), 0.0)
        facecolors = cmap(normed[:, :, k])
        facecolors[..., 3] = alpha
        Z = np.full_like(X, zc[k], dtype=float)
        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

    if projection_walls:
        xy = np.nansum(mag, axis=2)
        xz = np.nansum(mag, axis=1)
        yz = np.nansum(mag, axis=0)
        levels = np.geomspace(lo, max(hi, lo * 10.0), 32)
        XY_X, XY_Y = np.meshgrid(xc, yc, indexing="ij")
        XZ_X, XZ_Z = np.meshgrid(xc, zc, indexing="ij")
        YZ_Y, YZ_Z = np.meshgrid(yc, zc, indexing="ij")
        ax.contourf(XY_X, XY_Y, xy, zdir="z", offset=edges[2][0], levels=levels, cmap=cmap, alpha=0.22)
        ax.contourf(XZ_X, XZ_Z, xz, zdir="y", offset=edges[1][-1], levels=levels, cmap=cmap, alpha=0.18)
        ax.contourf(YZ_Y, YZ_Z, yz, zdir="x", offset=edges[0][0], levels=levels, cmap=cmap, alpha=0.18)

    ax.set_xlabel("x [M]")
    ax.set_ylabel("y [M]")
    ax.set_zlabel("z [M]")
    ax.set_xlim(edges[0][0], edges[0][-1])
    ax.set_ylim(edges[1][0], edges[1][-1])
    ax.set_zlim(edges[2][0], edges[2][-1])
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((
        edges[0][-1] - edges[0][0],
        edges[1][-1] - edges[1][0],
        edges[2][-1] - edges[2][0],
    ))
    ax.set_title(
        f"{title}\nbright transparent z-slices, threshold >= p{percentile:g}, slices={len(slice_ids)}"
    )
    mappable = ScalarMappable(norm=LogNorm(vmin=lo, vmax=hi), cmap=cmap)
    mappable.set_array([])
    cb = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
    cb.set_label("voxel value")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_3d_volume(
    output,
    hist,
    edges,
    title,
    max_voxels,
    percentile,
    cmap_name,
    style,
    display_sigma,
    max_slices,
    alpha_power,
    max_alpha,
    projection_walls,
):
    if style == "voxels":
        plot_3d_volume_voxels(output, hist, edges, title, max_voxels, percentile, cmap_name, display_sigma)
    else:
        plot_3d_volume_slices(
            output,
            hist,
            edges,
            title,
            max_slices,
            percentile,
            cmap_name,
            display_sigma,
            alpha_power,
            max_alpha,
            projection_walls,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--projection", choices=("rtheta", "cartesian", "both"), default="rtheta",
                        help="Default rtheta bins by the sample r,theta. Use cartesian/both for legacy x-y/x-z/3D projections.")
    parser.add_argument("--extent", type=float, default=40.0, help="Fallback half-width in M for legacy Cartesian axes and r max.")
    parser.add_argument("--r-range", default=None, help="BH-coordinate radial range, e.g. 0,20.")
    parser.add_argument("--theta-range", default=None, help="BH-coordinate theta range in radians, e.g. 0,pi.")
    parser.add_argument("--x-range", default=None, help="Legacy Cartesian x range, e.g. 0,10.")
    parser.add_argument("--y-range", default=None, help="Legacy Cartesian y range, e.g. -10,10.")
    parser.add_argument("--z-range", default=None, help="Legacy Cartesian z range, e.g. -10,10.")
    parser.add_argument("--xy-z-range", default=None, help="Optional z slab used only for legacy x-y projections.")
    parser.add_argument("--xz-y-range", default=None, help="Optional y slab used only for legacy x-z projections.")
    parser.add_argument("--bins", type=int, default=96, help="Number of bins per 2D axis. The r-theta default is intentionally coarser than the old Cartesian plots to avoid sparse point-like maps.")
    parser.add_argument("--smooth-sigma", type=float, default=0.0, help="Optional Gaussian smoothing sigma in 2D bin units. Default 0 keeps exact unsmoothed bins.")
    parser.add_argument("--bins3d", type=int, default=48)
    parser.add_argument("--panels", default=",".join(DEFAULT_PANELS), help="Comma-separated fields. Use ne_mean for dl-weighted density, ne_cgs_column for int ne_cgs dl, and dI/dQ/dU/dV for Stokes differences from recorded stokes samples.")
    parser.add_argument("--scalar3d", default="jI,ne_cgs_column", help="Comma-separated legacy 3D fields, e.g. jI,ne_cgs_column.")
    parser.add_argument("--volume-percentile", type=float, default=92.0)
    parser.add_argument("--max-3d-voxels", type=int, default=9000)
    parser.add_argument("--max-3d-points", type=int, default=None, help="Backward-compatible alias for --max-3d-voxels.")
    parser.add_argument("--volume-style", choices=("slices", "voxels"), default="slices",
                        help="3D rendering style. slices is smoother for presentation; voxels shows raw filled cells.")
    parser.add_argument("--volume-display-sigma", type=float, default=0.75,
                        help="Display-only Gaussian sigma in 3D voxel units. Raw binned NPZ output is unchanged.")
    parser.add_argument("--max-3d-slices", type=int, default=36)
    parser.add_argument("--volume-alpha-power", type=float, default=2.0,
                        help="Display-only alpha exponent for 3D slices. Larger values hide dim material and emphasize bright emission.")
    parser.add_argument("--volume-max-alpha", type=float, default=0.62,
                        help="Maximum slice opacity for --volume-style=slices.")
    parser.add_argument("--volume-projection-walls", action="store_true",
                        help="Draw projected wall contours behind the transparent slices. Disabled by default to avoid hiding bright structures.")
    parser.add_argument("--freq-index", type=int, default=0,
                        help="Frequency group to use for frequency-dependent fields in multi-frequency trace files.")
    args = parser.parse_args()

    r_range = parse_range(args.r_range, (0.0, args.extent))
    theta_range = parse_range(args.theta_range, (0.0, float(np.pi)))
    axis_default = (-args.extent, args.extent)
    x_range = parse_range(args.x_range, axis_default)
    y_range = parse_range(args.y_range, axis_default)
    z_range = parse_range(args.z_range, axis_default)
    xy_z_range = parse_range(args.xy_z_range, z_range) if args.xy_z_range is not None else None
    xz_y_range = parse_range(args.xz_y_range, y_range) if args.xz_y_range is not None else None
    max_voxels = args.max_3d_points if args.max_3d_points is not None else args.max_3d_voxels

    prefix = args.output_prefix or args.input.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as h5:
        layout = "ragged" if "/rays/sample_offset" in h5 else str(attr(h5, "trace_layout", "dense"))
        model = attr(h5, "model", "unknown")
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
        trace_fields = attr(h5, "trace_fields", "")
        trace_stride = int(attr(h5, "trace_stride", 1))
        counts = clipped_counts(h5)
        r = trace_field_flat(h5, "r", counts, args.freq_index)
        theta = trace_field_flat(h5, "theta", counts, args.freq_index)
        phi = trace_field_flat(h5, "phi", counts, args.freq_index)
        if r is None or theta is None or phi is None:
            raise RuntimeError("trace requires coords fields: r, theta, phi")
        r = r.astype(float)
        theta = theta.astype(float)
        phi = phi.astype(float)
        dlambda = trace_field_flat(h5, "dlambda", counts, args.freq_index)
        if dlambda is None:
            weight = np.ones_like(r, dtype=float)
            weight_label = "sample count"
        else:
            weight = np.abs(dlambda.astype(float))
            weight_label = "|dlambda|"
        requested = {p.strip() for p in args.panels.split(",") if p.strip()}
        if args.projection in {"cartesian", "both"}:
            requested.update(p.strip() for p in args.scalar3d.split(",") if p.strip())
        fields = {}
        for field in sorted(requested):
            base, _mode = field_base(field)
            if base == "path_length" or base in fields:
                continue
            if base in DELTA_FIELDS:
                arr = trace_delta_field_flat(h5, base, counts, args.freq_index)
            else:
                arr = trace_field_flat(h5, base, counts, args.freq_index)
            if arr is not None:
                fields[base] = arr.astype(float)

    delta_requested = any(is_delta_panel(p.strip()) for p in args.panels.split(",") if p.strip())
    delta_available = any(name in fields for name in DELTA_FIELDS)
    if delta_requested and not delta_available:
        print("warning: dI/dQ/dU/dV require trace_fields including stokes; delta panels will be skipped", file=sys.stderr)
    if delta_available and trace_stride != 1:
        print(f"warning: Stokes delta fields are differences between recorded samples with trace_stride={trace_stride}; use trace_stride=1 for per-step localization", file=sys.stderr)

    r_mid = midpoint_field_flat(r, counts)
    theta_mid = midpoint_field_flat(theta, counts)
    phi_mid = midpoint_angle_flat(phi, counts)

    base_mask = finite_mask(r, theta, phi, weight) & (weight > 0) & (r >= 0)
    r, theta, phi = r[base_mask], theta[base_mask], phi[base_mask]
    r_mid, theta_mid, phi_mid = r_mid[base_mask], theta_mid[base_mask], phi_mid[base_mask]
    weight = weight[base_mask]
    for name in list(fields):
        fields[name] = fields[name][base_mask]

    panels = [p.strip() for p in args.panels.split(",") if p.strip()]
    freq_label = f", freq_index={args.freq_index}" + (f", nu={frequency:.6g} Hz" if frequency is not None else "")
    title_base = f"BH-coordinate trace bins: model={model}, layout={layout}{freq_label}, fields={trace_fields}, weight={weight_label}"
    written = []

    if args.projection in {"rtheta", "both"}:
        rtheta_maps = []
        for panel in panels:
            panel_r = r_mid if is_delta_panel(panel) else r
            panel_theta = theta_mid if is_delta_panel(panel) else theta
            rt = make_map_with_edges(panel, panel_r, panel_theta, fields, weight, r_range, theta_range, args.bins, args.smooth_sigma)
            if rt is not None:
                rtheta_maps.append(rt)
        rtheta_path = prefix.with_name(prefix.name + "_rtheta.png")
        meridional_path = prefix.with_name(prefix.name + "_meridional.png")
        title_rt = title_base + f" (r-theta bins, r in [{r_range[0]}, {r_range[1]}] M, smooth_sigma={args.smooth_sigma:g})"
        interp = "bilinear" if args.smooth_sigma > 0 else "nearest"
        plot_rtheta_projection(rtheta_path, rtheta_maps, r_range, theta_range, title_rt, interp)
        plot_meridional_projection(meridional_path, rtheta_maps, title_rt + " mirrored in cylindrical R")
        written.extend([rtheta_path, meridional_path])

    if args.projection in {"cartesian", "both"}:
        x, y, z = cartesian_from_spherical(r, theta, phi)
        x_mid, y_mid, z_mid = cartesian_from_spherical(r_mid, theta_mid, phi_mid)
        xy_mask = np.ones_like(x, dtype=bool) if xy_z_range is None else range_mask(z, xy_z_range)
        xz_mask = np.ones_like(x, dtype=bool) if xz_y_range is None else range_mask(y, xz_y_range)
        xy_mid_mask = np.ones_like(x_mid, dtype=bool) if xy_z_range is None else range_mask(z_mid, xy_z_range)
        xz_mid_mask = np.ones_like(x_mid, dtype=bool) if xz_y_range is None else range_mask(y_mid, xz_y_range)
        xy_maps = []
        xz_maps = []
        for panel in panels:
            if is_delta_panel(panel):
                xy = make_map(panel, x_mid[xy_mid_mask], y_mid[xy_mid_mask], {k: v[xy_mid_mask] for k, v in fields.items()}, weight[xy_mid_mask], x_range, y_range, args.bins, args.smooth_sigma)
                xz = make_map(panel, x_mid[xz_mid_mask], z_mid[xz_mid_mask], {k: v[xz_mid_mask] for k, v in fields.items()}, weight[xz_mid_mask], x_range, z_range, args.bins, args.smooth_sigma)
            else:
                xy = make_map(panel, x[xy_mask], y[xy_mask], {k: v[xy_mask] for k, v in fields.items()}, weight[xy_mask], x_range, y_range, args.bins, args.smooth_sigma)
                xz = make_map(panel, x[xz_mask], z[xz_mask], {k: v[xz_mask] for k, v in fields.items()}, weight[xz_mask], x_range, z_range, args.bins, args.smooth_sigma)
            if xy is not None:
                xy_maps.append(xy)
            if xz is not None:
                xz_maps.append(xz)
        if xy_z_range is not None:
            title_xy = title_base + f" (legacy x-y, z in [{xy_z_range[0]}, {xy_z_range[1]}] M)"
        else:
            title_xy = title_base + " (legacy x-y, projected over selected z range)"
        if xz_y_range is not None:
            title_xz = title_base + f" (legacy x-z, y in [{xz_y_range[0]}, {xz_y_range[1]}] M)"
        else:
            title_xz = title_base + " (legacy x-z, projected over selected y range)"
        xy_path = prefix.with_name(prefix.name + "_xy.png")
        xz_path = prefix.with_name(prefix.name + "_xz.png")
        plot_projection(xy_path, xy_maps, x_range, y_range, title_xy, "x [M]", "y [M]")
        plot_projection(xz_path, xz_maps, x_range, z_range, title_xz, "x [M]", "z [M]")
        written.extend([xy_path, xz_path])

        volume_mask = range_mask(x, x_range) & range_mask(y, y_range) & range_mask(z, z_range)
        scalar3d_list = [p.strip() for p in args.scalar3d.split(",") if p.strip()]
        for scalar in scalar3d_list:
            hist3d, edges3d, scalar_title, _scale = hist3d_scalar(x[volume_mask], y[volume_mask], z[volume_mask], scalar, {k: v[volume_mask] for k, v in fields.items()}, weight[volume_mask], (x_range, y_range, z_range), args.bins3d)
            suffix = scalar.replace("/", "_")
            out3d = prefix.with_name(prefix.name + f"_3d_{suffix}.png")
            cmap = "inferno" if scalar.startswith("j") else "viridis"
            plot_3d_volume(
                out3d,
                hist3d,
                edges3d,
                f"{model}: {scalar_title}",
                max_voxels,
                args.volume_percentile,
                cmap,
                args.volume_style,
                args.volume_display_sigma,
                args.max_3d_slices,
                args.volume_alpha_power,
                args.volume_max_alpha,
                args.volume_projection_walls,
            )
            np.savez_compressed(
                prefix.with_name(prefix.name + f"_binned_{suffix}.npz"),
                hist3d=hist3d,
                x_edges=edges3d[0],
                y_edges=edges3d[1],
                z_edges=edges3d[2],
                scalar3d=scalar,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                bins=args.bins,
                bins3d=args.bins3d,
            )
            written.append(out3d)

    print(f"samples_loaded {r.size}")
    print(f"samples_in_rtheta {np.count_nonzero(range_mask(r, r_range) & range_mask(theta, theta_range))}")
    print(f"fields_available {','.join(sorted(fields))}")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
