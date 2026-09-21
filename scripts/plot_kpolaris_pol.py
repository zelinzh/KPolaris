#!/usr/bin/env python3
"""Plot a selected native Stokes image or polarization summary on a common scale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kpolaris_evpa import qu_basis_sign
from kpolaris_image import image_extent, image_summary, load_image, evpa_tick_vectors


def polarization_maps(stokes, valid, floor):
    """Mask fractional/angle displays; never change or renormalize stored Stokes."""
    i, q, u, v = stokes
    peak = float(np.max(i[valid])) if valid.any() else 0.
    bright = valid & (i > max(0., peak * floor))
    linear = np.hypot(q, u)
    lp = np.divide(100 * linear, i, out=np.full_like(i, np.nan), where=bright)
    cp = np.divide(100 * v, i, out=np.full_like(i, np.nan), where=bright)
    angle = np.where(bright & (linear > peak * floor), np.degrees(.5 * np.arctan2(u, q)), np.nan)
    return lp, angle, cp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--freq-index", type=int, default=0)
    parser.add_argument("--layout", choices=("intensity", "polarization", "stokes"), default="polarization")
    parser.add_argument("--columns", type=int, choices=(2, 4), default=2,
                        help="panel columns for stokes/polarization layouts (default: 2)")
    parser.add_argument("--stokes-scale", choices=("shared", "independent"), default="shared",
                        help="shared uses peak I for all Stokes limits; independent uses each component's peak amplitude")
    parser.add_argument("--intensity-unit", choices=("cgs", "brightness-temperature"), default="cgs",
                        help="linear specific intensity, or Rayleigh-Jeans brightness temperature in 10^9 K")
    parser.add_argument("--title", help="optional figure title; does not change the image or its orientation")
    parser.add_argument("--fov-units", choices=("muas", "M", "pixel"), default="M")
    parser.add_argument("--distance-pc", type=float, help="explicit angular-scale override")
    parser.add_argument("--mass-solar", type=float, help="explicit angular-scale override")
    parser.add_argument("--scale", type=float, help="explicit cgs intensity to Jy/pixel conversion")
    parser.add_argument("--qu-conv", choices=("stored", "camera"), default="stored",
                        help="stored preserves the file N/W basis; camera explicitly converts to the internal W basis")
    parser.add_argument("--evpa-conv", choices=("stored", "camera"), default="stored",
                        help="stored uses the recorded evpa_0; camera explicitly converts Q/U and EVPA to W")
    parser.add_argument("--evpa-ticks", action="store_true", help="equal-length polarization segments in the camera basis")
    parser.add_argument("--intensity-floor", type=float, default=.001,
                        help="fraction of peak I below which fractional/angle displays are masked (default .001)")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    if args.scale is not None and args.intensity_unit != "cgs":
        parser.error("--scale and brightness-temperature select different display units; choose one")
    if not 0 <= args.intensity_floor < 1 or args.dpi <= 0:
        parser.error("intensity-floor must be in [0,1) and dpi must be positive")
    for name in ("scale", "distance_pc", "mass_solar"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    output = args.output or args.input.with_suffix(".png")
    if output.resolve() == args.input.resolve():
        parser.error("output must differ from the input HDF5 file")
    if args.summary_json and args.summary_json.resolve() in (args.input.resolve(), output.resolve()):
        parser.error("summary path must differ from input and figure paths")
    try:
        image = load_image(args.input, args.frame, args.freq_index)
        extent, axis_unit = image_extent(image, args.fov_units, args.distance_pc, args.mass_solar)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.qu_conv == "camera" or args.evpa_conv == "camera":
        image["stokes"][1:3] *= qu_basis_sign(image["attrs"]["evpa_0"], "W")
        image["attrs"]["evpa_0"] = "W"
    if args.scale is not None:
        image["attrs"]["intensity_to_flux_jy_per_pixel"] = args.scale
        image["attrs"]["flux_scale_valid"] = 1
    summary = image_summary(image)
    summary.update({"qu_convention": args.qu_conv, "evpa_convention": args.evpa_conv,
                    "display_intensity_floor": args.intensity_floor, "layout": args.layout,
                    "display_columns": 1 if args.layout == "intensity" else args.columns,
                    "display_stokes_scale": args.stokes_scale,
                    "display_intensity_unit": "Jy/pixel" if args.scale is not None else args.intensity_unit})
    i, q, u, v = image["stokes"]
    valid = image["valid"]
    lp, angle, cp = polarization_maps(image["stokes"], valid, args.intensity_floor)
    if args.evpa_ticks:
        try:
            vx, vy = evpa_tick_vectors(image, np.radians(angle), args.fov_units)
        except ValueError as exc:
            parser.error(str(exc))
    peak = max(float(np.max(i[valid])) if valid.any() else 0., np.finfo(float).tiny)
    if args.layout == "intensity":
        arrays = [np.where(valid, i, np.nan)]
        limits = [(0, peak)]
        cmaps = ["afmhot"]
        titles = ["Stokes I"]
        labels = [r"$I_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]"]
    elif args.layout == "stokes":
        arrays = [np.where(valid, a, np.nan) for a in (i, q, u, v)]
        limits = [(0, peak)] + [(-peak, peak)] * 3
        if args.stokes_scale == "independent":
            for idx, values in enumerate((q, u, v), start=1):
                amplitude = float(np.max(np.abs(values[valid]))) if valid.any() else 0.
                amplitude = amplitude if amplitude > 0 else peak
                limits[idx] = (-amplitude, amplitude)
        cmaps = ["magma", "RdBu_r", "RdBu_r", "RdBu_r"]
        titles = [f"Stokes {s}" for s in "IQUV"]
        labels = [rf"${s}_\nu$ [erg s$^{{-1}}$ cm$^{{-2}}$ Hz$^{{-1}}$ sr$^{{-1}}$]" for s in "IQUV"]
    else:
        cmax = max(float(np.nanmax(np.abs(cp))) if np.isfinite(cp).any() else 1., 1e-8)
        arrays = [np.where(valid, i, np.nan), lp, angle, cp]
        lmax = max(100., float(np.nanmax(lp)) if np.isfinite(lp).any() else 100.)
        limits = [(0, peak), (0, lmax), (-90, 90), (-cmax, cmax)]
        cmaps = ["magma", "cividis", "twilight_shifted", "RdBu_r"]
        titles = ["Stokes I", "Linear polarization", f"EVPA ({image['attrs']['evpa_0']} zero)", "Circular polarization"]
        labels = [r"$I_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]", r"$100\,\sqrt{Q^2+U^2}/I$ [%]", r"$\chi$ [deg]", r"$100\,V/I$ [%]"]
    if args.scale is not None:
        for idx in (range(4) if args.layout == "stokes" else (0,)):
            arrays[idx] = arrays[idx] * args.scale
            limits[idx] = tuple(v * args.scale for v in limits[idx])
            labels[idx] = f"Stokes {'IQUV'[idx]} [Jy / pixel]"
    elif args.intensity_unit == "brightness-temperature":
        hz = summary["frequency_hz"]
        if hz is None or hz <= 0:
            parser.error("brightness temperature requires a positive frequency in the selected image")
        # Rayleigh-Jeans brightness temperature: T_b = c^2 I_nu / (2 k_B nu^2).
        # This is a unit conversion, not the local electron temperature.
        factor = 2.99792458e10**2 / (2 * 1.380649e-16 * hz**2) / 1e9
        for idx in (range(4) if args.layout == "stokes" else (0,)):
            arrays[idx] = arrays[idx] * factor
            limits[idx] = tuple(v * factor for v in limits[idx])
            labels[idx] = ("Brightness temperature" if idx == 0 else f"Stokes {'IQUV'[idx]}") + r" [$10^9$ K]"
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.facecolor": "white"}):
        single = args.layout == "intensity"
        strip = not single and args.columns == 4
        fig, axes = plt.subplots(1 if single or strip else 2, 1 if single else args.columns,
                                 figsize=(6.4, 5.6) if single else (12.0, 3.7) if strip else (9.2, 8.2),
                                 squeeze=False, layout="constrained")
        for ax, values, (lo, hi), cmap, title, label in zip(axes.flat, arrays, limits, cmaps, titles, labels):
            colors = plt.get_cmap(cmap).copy(); colors.set_bad("#e7e7e7")
            im = ax.imshow(values, origin="lower", extent=extent, interpolation="nearest", cmap=colors, vmin=lo, vmax=hi)
            ax.set(title="" if single else title, xlabel=f"Image x [{axis_unit}]", ylabel=f"Image y [{axis_unit}]")
            bar = fig.colorbar(im, ax=ax, orientation="horizontal" if strip else "vertical",
                               fraction=.055 if strip else .046, pad=.08 if strip else .035)
            bar.set_label(label)
            if title.startswith("EVPA"):
                bar.set_ticks([-90, -45, 0, 45, 90])
            ax.set_aspect("equal")
        if args.evpa_ticks:
            ny, nx = i.shape
            xs = extent[0] + (np.arange(nx) + .5) * (extent[1] - extent[0]) / nx
            ys = extent[2] + (np.arange(ny) + .5) * (extent[3] - extent[2]) / ny
            x, y = np.meshgrid(xs, ys)
            skip = max(1, max(nx, ny) // 20)
            keep = np.isfinite(angle[::skip, ::skip])
            axes[0, 0].quiver(x[::skip, ::skip][keep], y[::skip, ::skip][keep], vx[::skip, ::skip][keep], vy[::skip, ::skip][keep],
                              angles="xy", pivot="middle", headwidth=0, headlength=0, headaxislength=0,
                              color="white", width=.003, scale=35)
        hz = summary["frequency_hz"]
        metadata = (f"{hz / 1e9:g} GHz  ·  " if hz else "") + f"{i.shape[1]} × {i.shape[0]}"
        if args.frame:
            metadata += f"  ·  frame {args.frame}"
        fig.suptitle((args.title + "\n" if args.title else "") + metadata, fontsize=12)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=args.dpi)
        plt.close(fig)
    flux = summary["flux_valid_pixels_jy"]
    if flux is not None:
        print(f"Flux [Jy]:    {flux['I']:g}")
        print("I,Q,U,V [Jy]: " + " ".join(f"{flux[s]:g}" for s in "IQUV"))
    else:
        print("Flux [Jy]: unavailable (no valid physical flux scale in the input)")
    print(f"Valid pixels: {summary['valid_pixels']}/{summary['pixels']}")
    if not summary["ok"]:
        print("Only finite, returned rays are displayed; inspect termination_counts before using this image.", file=sys.stderr)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
