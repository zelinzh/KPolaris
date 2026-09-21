#!/usr/bin/env python3
"""Observer-basis polarization cancellation and fixed-operator source response.

Uses existing analysis-v2 radial Stokes cubes; no ray integration or fitting.
Coherence refers to the recorded radial partition, not a Faraday-only effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

STOKES = "IQUV"


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def radial_amplitude_maps(analysis, scale=1.0):
    """Read bins individually, including pixels with exactly cancelling Stokes."""
    q = analysis["radial_stokes_Q_inv_contribution"]
    u = analysis["radial_stokes_U_inv_contribution"]
    v = analysis["radial_stokes_V_inv_contribution"]
    if q.ndim != 3 or q.shape != u.shape or q.shape != v.shape:
        raise ValueError("radial Q/U/V cubes must have matching (bin, y, x) shapes")
    linear = np.zeros(q.shape[1:])
    circular = np.zeros_like(linear)
    for k in range(q.shape[0]):
        linear += np.hypot(np.asarray(q[k]) * scale, np.asarray(u[k]) * scale)
        circular += np.abs(np.asarray(v[k]) * scale)
    if not np.isfinite(linear).all() or not np.isfinite(circular).all():
        raise ValueError("non-finite radial polarization contribution")
    return linear, circular


def component_budget(radial, observed, circular=False):
    tagged = float(np.sum(np.abs(radial)))
    resolved = float(np.sum(np.abs(observed)))
    net = float(abs(np.sum(observed)))
    return {
        "tagged_amplitude": tagged,
        "resolved_amplitude": resolved,
        "net_amplitude": net,
        "los_coherence": ratio(resolved, tagged),
        "image_coherence": ratio(net, resolved),
        "total_coherence": ratio(net, tagged),
        "los_cancellation_fraction": ratio(tagged - resolved, tagged),
        "image_cancellation_fraction": ratio(resolved - net, tagged),
        "surviving_fraction": ratio(net, tagged),
        "signed_net": float(np.sum(observed)) if circular else None,
    }


def cancellation_budget(stokes, radial, closure_limit=1e-10, polarization_floor=1e-10):
    """Arrays have shape (4, pixel) and (4, bin, pixel); equal solid-angle pixels."""
    stokes = np.asarray(stokes, dtype=float)
    radial = np.asarray(radial, dtype=float)
    if stokes.ndim != 2 or radial.ndim != 3 or stokes.shape[0] != 4 or radial.shape[0] != 4:
        raise ValueError("expected (4,pixel) Stokes and (4,bin,pixel) source tags")
    if stokes.shape[1] != radial.shape[2] or radial.shape[1] == 0 or stokes.shape[1] == 0:
        raise ValueError("empty or inconsistent pixel/bin dimensions")
    if not np.isfinite(stokes).all() or not np.isfinite(radial).all():
        raise ValueError("non-finite Stokes/source tag")
    # Ratios and response coefficients are independent of physical intensity units.
    scale = max(float(np.max(np.abs(stokes))), float(np.max(np.abs(radial))))
    if scale == 0:
        raise ValueError("no radiation: polarization budgets are undefined")
    stokes = stokes / scale
    radial = radial / scale
    closure_denominator = np.sum(np.abs(radial), axis=(0, 1))
    closure_residual = np.sum(np.abs(np.sum(radial, axis=1) - stokes), axis=0)
    closure = np.divide(closure_residual, closure_denominator,
                        out=np.where(closure_residual == 0, 0.0, np.inf),
                        where=closure_denominator > 0)
    if np.max(closure) > closure_limit:
        raise ValueError(f"radial Stokes closure failed: {np.max(closure):.6g}")
    total_i = float(np.sum(stokes[0]))
    if total_i <= 0 or np.min(stokes[0]) < -closure_limit or np.min(radial[0]) < -closure_limit:
        raise ValueError("positive emitted/observed intensity required")
    p = stokes[1] + 1j * stokes[2]
    pk = radial[1] + 1j * radial[2]
    linear = component_budget(pk, p)
    circular = component_budget(radial[3], stokes[3], circular=True)
    for budget in (linear, circular):
        for name in ("tagged", "resolved", "net"):
            budget[f"{name}_fraction"] = budget[f"{name}_amplitude"] / total_i
        budget["factorization_residual"] = (
            budget["tagged_fraction"] * budget["total_coherence"] - budget["net_fraction"]
            if budget["total_coherence"] is not None else None)
        # Normalized numbers avoid dependence on invariant-intensity units.
        for name in ("tagged_amplitude", "resolved_amplitude", "net_amplitude", "signed_net"):
            budget.pop(name)
    nr = radial.shape[1]
    coarsening = []
    factors = sorted({1, nr} | {2**k for k in range(1, nr.bit_length()) if nr % 2**k == 0})
    for factor in factors:
        grouped = radial.reshape(4, nr // factor, factor, -1).sum(axis=2)
        row = {"merge_factor": factor, "bins": nr // factor}
        for name, tags, final, circ in (
            ("linear", grouped[1] + 1j * grouped[2], p, False),
            ("circular", grouped[3], stokes[3], True),
        ):
            b = component_budget(tags, final, circ)
            row[name + "_los_coherence"] = b["los_coherence"]
            row[name + "_tagged_fraction"] = b["tagged_amplitude"] / total_i
        coarsening.append(row)
    integrated = radial.sum(axis=2)
    ptot = np.sum(p)
    pk_integrated = integrated[1] + 1j * integrated[2]
    p_scale = float(np.sum(np.abs(pk)))
    response_valid = abs(ptot) > polarization_floor * p_scale
    net_l = float(abs(ptot)) / total_i
    net_c = float(np.sum(stokes[3])) / total_i
    responses = []
    for k in range(nr):
        ik, qk, uk, vk = integrated[:, k]
        z = pk_integrated[k] / ptot if response_valid else None
        responses.append({
            "bin": k,
            "dln_flux_I_d_emission_scale": float(ik / total_i),
            "dln_net_linear_amplitude_d_emission_scale": float(z.real) if z is not None else None,
            "d_net_evpa_rad_d_emission_scale": float(0.5 * z.imag) if z is not None else None,
            "d_net_linear_fraction_d_emission_scale": float(net_l * (z.real - ik / total_i)) if z is not None else None,
            "d_net_circular_fraction_d_emission_scale": float((vk - net_c * ik) / total_i),
            "integrated_stokes_over_total_I": [float(x / total_i) for x in (ik, qk, uk, vk)],
        })
    return {
        "closure_max_relative_l1": float(np.max(closure)),
        "linear": linear, "circular": circular,
        "net_circular_fraction_signed": net_c,
        "radial_coarsening": coarsening,
        "source_response_linear_valid": bool(response_valid),
        "source_response_polarization_floor": polarization_floor,
        "source_response": responses,
    }


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def analyze_image(path, freq_index=0, closure_limit=1e-10, polarization_floor=1e-10):
    with h5py.File(path, "r") as h5:
        frame = h5["frame_0"]
        freq = frame[f"freq_{freq_index}"]
        if "analysis" in freq:
            analysis = freq["analysis"]
        elif int(h5.attrs.get("nfreq", 1)) == 1 and "analysis" in frame:
            analysis = frame["analysis"]
        else:
            raise ValueError("selected frequency has no independent analysis group")
        if int(analysis.attrs.get("observer_weighted_available", 0)) != 1:
            raise ValueError("complete observer-weighted analysis v2 is required")
        stokes = np.stack([freq[f"{s}_inv"][...] for s in STOKES])
        radial = np.stack([analysis[f"radial_stokes_{s}_inv_contribution"][...] for s in STOKES])
        reason = np.asarray(frame["diagnostics/reason"])
        if stokes.shape[1:] != reason.shape or radial.shape[2:] != reason.shape:
            raise ValueError("Stokes/source-tag/termination-map shape mismatch")
        valid = reason == 1
        edges = np.asarray(analysis["radial_bin_edges"])
        if len(edges) != radial.shape[1] + 1 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
            raise ValueError("invalid radial bin edges")
        budget = cancellation_budget(stokes[:, valid], radial[:, :, valid], closure_limit, polarization_floor)
        budget.update({
            "input": str(Path(path).resolve()), "input_sha256": sha256(path),
            "source_revision": str(h5.attrs.get("code_revision", "unknown")),
            "source_fingerprint": str(h5.attrs.get("code_source_fingerprint", "unknown")),
            "model": str(h5.attrs.get("model", "unknown")),
            "frequency_hz": float(freq.attrs.get("frequency_hz", h5.attrs.get("frequency_hz", 0))),
            "freq_index": freq_index, "shape": list(reason.shape),
            "valid_pixels": int(valid.sum()), "excluded_pixels": int((~valid).sum()),
            "all_rays_returned": bool(valid.all()),
            "radial_bin_edges_M": edges.tolist(),
            "radial_edge_policy": "first/last bins include emission outside configured radial bounds",
        })
    return budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--freq-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--closure-limit", type=float, default=1e-10)
    parser.add_argument("--polarization-floor", type=float, default=1e-10)
    args = parser.parse_args()
    if not 0 < args.closure_limit < 1 or not 0 < args.polarization_floor < 1:
        parser.error("closure limit and polarization floor must lie between zero and one")
    payload = {
        "schema": "kpolaris_polarization_budget", "schema_version": 1,
        "script_sha256": sha256(__file__),
        "definitions": {
            "weights": "equal-solid-angle pixels; observer-basis invariant Stokes; reason=1 mask; no intensity cut",
            "cancellation": "ordered sums over radial bins then image pixels; bin-dependent, not a Faraday-only attribution",
            "source_response": "derivative for j_k -> (1+epsilon_k) j_k at fixed ray, plasma transfer operator K and other source bins; not a density/heating derivative",
            "coarsening": "merge existing adjacent source tags; does not establish finer-bin or image-resolution convergence",
            "undefined": "zero-denominator coherence and ill-conditioned net EVPA response are null",
        },
        "images": [analyze_image(p, args.freq_index, args.closure_limit, args.polarization_floor) for p in args.images],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(payload['images'])} polarization budgets to {args.output}")


if __name__ == "__main__":
    main()
