#!/usr/bin/env python3
"""Compare complete slow-light batches and summarize exclusive driver timers.

NMSE is sum((candidate-reference)^2)/sum(reference^2), independently for IQUV.
The input/service timers may overlap kernels and are intentionally not added to
one another. A different tolerance is allowed and explicitly recorded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import h5py
import numpy as np
from kpolaris_evpa import evpa_zero, qu_basis_sign


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metric(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('shape mismatch or nonfinite image data')
    d = b - a
    denominator = float(np.sum(a * a))
    numerator = float(np.sum(d * d))
    nmse = numerator / denominator if denominator else (0.0 if not numerator else None)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    maximum = float(np.max(np.abs(d))) if d.size else 0.0
    return dict(nmse=nmse, relative_l2=math.sqrt(nmse) if nmse is not None else None,
                max_abs=maximum, max_abs_over_reference_peak=maximum / peak if peak else None,
                bitwise_equal=bool(np.array_equal(a, b)))


def timings(path):
    if not path or not path.exists():
        return None
    result = {}
    for label, seconds in re.findall(r'^timing (\w+) ([0-9.eE+\-]+) s\s*$', path.read_text(), re.M):
        entry = result.setdefault(label, dict(calls=0, seconds=0.0))
        entry['calls'] += 1
        entry['seconds'] += float(seconds)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.count < 1:
        p.error('--count must be positive')
    rows = []
    # Keep the physical comparison fixed. Tolerance is the only numerical
    # setting that this report permits to differ without an explicit failure.
    keys = ['nx', 'ny', 'model', 'coordinate', 'camera', 'spin', 'camera_radius',
            'inclination_rad', 'fov', 'fovy', 'inner_radius', 'outer_radius', 'frequency_hz',
            'slow_light_observation_time', 'slow_light_time_list', 'direct_only',
            'equatorial_h_over_r', 'faraday_rotation', 'step', 'max_step', 'min_step',
            'max_steps', 'max_radiation_step', 'max_radiation_depth',
            'max_absorption_depth', 'max_faraday_depth', 'analysis_mode',
            'kharma_M_unit', 'kharma_trat_small', 'kharma_trat_large', 'kharma_sigma_cut']
    for i in range(args.count):
        name = f'frame_{i:05d}.h5'
        ref, cand = args.reference / name, args.candidate / name
        with h5py.File(ref) as a, h5py.File(cand) as b:
            for key in keys:
                if key not in a.attrs or key not in b.attrs or a.attrs[key] != b.attrs[key]:
                    raise ValueError(f'{name}: mismatched/missing comparison parameter {key}')
            reasons = b['frame_0/diagnostics/reason'][...]
            if not np.all(reasons == 1):
                raise ValueError(f'{name}: rays did not return to camera: {np.unique(reasons, return_counts=True)}')
            az = evpa_zero(a.attrs, a['frame_0/freq_0'].attrs)
            bz = evpa_zero(b.attrs, b['frame_0/freq_0'].attrs)
            row = dict(reference_evpa_0=az, candidate_evpa_0=bz, comparison_evpa_0='N', frame=i, reference_sha256=sha256(ref), candidate_sha256=sha256(cand),
                       reference_tolerance=float(a.attrs['adaptive_tolerance']),
                       candidate_tolerance=float(b.attrs['adaptive_tolerance']),
                       stokes={c: metric(a[f'frame_0/freq_0/{c}'][...] * (qu_basis_sign(az, 'N') if c in 'QU' else 1), b[f'frame_0/freq_0/{c}'][...] * (qu_basis_sign(bz, 'N') if c in 'QU' else 1)) for c in 'IQUV'},
                       diagnostics={})
            for key in a['frame_0/diagnostics']:
                av, bv = a[f'frame_0/diagnostics/{key}'][...], b[f'frame_0/diagnostics/{key}'][...]
                row['diagnostics'][key] = metric(av, bv)
            row['steps_mean'] = float(np.mean(b['frame_0/diagnostics/steps'][...]))
            row['steps_max'] = int(np.max(b['frame_0/diagnostics/steps'][...]))
            rows.append(row)
    result = dict(reference=str(args.reference.resolve()), candidate=str(args.candidate.resolve()),
                  count=args.count, frames=rows,
                  reference_timings=timings(args.reference / 'batch.log'),
                  candidate_timings=timings(args.candidate / 'batch.log'),
                  max_nmse={c: (None if any(r['stokes'][c]['nmse'] is None for r in rows)
                                else max(r['stokes'][c]['nmse'] for r in rows)) for c in 'IQUV'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(count=args.count, max_nmse=result['max_nmse']), indent=2))


if __name__ == '__main__':
    main()
