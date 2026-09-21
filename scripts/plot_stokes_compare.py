#!/usr/bin/env python3
"""Plot two KPolaris-style Stokes CSV images and their differences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv_image(path: Path, columns: list[str]) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    metadata: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        header: list[str] | None = None
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
            rows.append(dict(zip(header, values)))
    if not rows:
        raise ValueError(f"{path}: no data rows found")
    nx = max(int(row["ix"]) for row in rows) + 1
    ny = max(int(row["iy"]) for row in rows) + 1
    arrays = {name: np.full((ny, nx), np.nan, dtype=float) for name in columns}
    for row in rows:
        ix = int(row["ix"])
        iy = int(row["iy"])
        for name in columns:
            arrays[name][iy, ix] = float(row[name])
    return metadata, arrays


def symmetric_limit(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    lim = float(np.percentile(np.abs(finite), percentile))
    return lim if lim > 0 else 1.0


def positive_limit(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(finite, percentile))
    return vmax if vmax > 0 else float(np.nanmax(finite)) if np.nanmax(finite) > 0 else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", default="I_nu,Q_nu,U_nu,V_nu")
    parser.add_argument("--reference-label", default="Reference")
    parser.add_argument("--candidate-label", default="KPolaris")
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if not 0 < args.percentile <= 100 or args.dpi <= 0:
        parser.error("percentile must be in (0,100] and dpi must be positive")
    columns = [item.strip() for item in args.columns.split(",") if item.strip()]
    if not columns:
        parser.error("select at least one Stokes column")
    _, ref = read_csv_image(args.reference, columns)
    _, got = read_csv_image(args.candidate, columns)

    if any(ref[name].shape != got[name].shape for name in columns):
        parser.error("reference and candidate image dimensions differ")
    ncols = len(columns)
    fig, axes = plt.subplots(3, ncols, figsize=(3.2 * ncols, 8.2), constrained_layout=True)
    if ncols == 1:
        axes = axes.reshape(3, 1)

    for c, name in enumerate(columns):
        ref_arr = ref[name]
        got_arr = got[name]
        diff = got_arr - ref_arr
        if name.startswith("I"):
            vmax = positive_limit(np.concatenate([ref_arr.ravel(), got_arr.ravel()]), args.percentile)
            cmap = "inferno"
            kwargs = dict(vmin=0.0, vmax=vmax, cmap=cmap, origin="lower")
        else:
            vmax = symmetric_limit(np.concatenate([ref_arr.ravel(), got_arr.ravel()]), args.percentile)
            kwargs = dict(vmin=-vmax, vmax=vmax, cmap="RdBu_r", origin="lower")
        dlim = symmetric_limit(diff, args.percentile)

        panels = [
            (axes[0, c], ref_arr, args.reference_label, kwargs),
            (axes[1, c], got_arr, args.candidate_label, kwargs),
            (axes[2, c], diff, f"{args.candidate_label} - {args.reference_label}", dict(vmin=-dlim, vmax=dlim, cmap="RdBu_r", origin="lower")),
        ]
        for ax, arr, label, im_kwargs in panels:
            im = ax.imshow(arr, **im_kwargs)
            ax.set_title(f"{label}\n{name}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.78)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
