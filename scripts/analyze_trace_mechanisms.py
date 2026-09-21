#!/usr/bin/env python3
"""Replay full traces and locate circular-polarization production mechanisms.

Independent matrix-exponential absorption/emission plus Rodrigues rotation
reproduces the archived Strang step. Controlled coefficient interventions and
refined replay distinguish mechanism effects from accumulated coefficient depth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_polarization_budget import sha256

CHANNELS = ('intrinsic_emission', 'selective_absorption', 'faraday_conversion', 'incident_boundary')


def rotation_step(s, rho, dl):
    norm = float(np.linalg.norm(rho))
    if norm == 0 or dl == 0:
        return s.copy()
    axis = rho / norm; theta = norm * dl
    out = s.copy(); p = s[1:]
    out[1:] = p * np.cos(theta) + axis * np.dot(axis, p) * (1 - np.cos(theta)) + np.cross(axis, p) * np.sin(theta)
    return out


def absorption_propagator(j, alpha, dl):
    # Independent eigendecomposition of the symmetric absorption matrix;
    # unlike the production code, no parallel/perpendicular mode formulas.
    matrix = -np.eye(4) * alpha[0]
    matrix[0, 1:] = -alpha[1:]
    matrix[1:, 0] = -alpha[1:]
    eigenvalues, vectors = np.linalg.eigh(matrix * dl)
    exponential = np.exp(eigenvalues)
    factor = np.empty(4)
    small = np.abs(eigenvalues) < 1e-8
    factor[small] = 1 + eigenvalues[small]/2 + eigenvalues[small]**2/6
    factor[~small] = np.expm1(eigenvalues[~small])/eigenvalues[~small]
    prop = np.eye(5)
    prop[:4,:4] = (vectors * exponential) @ vectors.T
    prop[:4,4] = ((vectors * factor) @ vectors.T) @ (j * dl)
    return prop


def replay(j, alpha, rho, dl, initial=None, substeps=1, rotation_scale=1., conversion_scale=1., split_index=None):
    """Piecewise-constant, prescribed plasma; return S and signed CP production.

Channels solve dV_c/dlambda = source_c - alpha_I V_c with sources jV,
-alphaV I, rhoQ U-rhoU Q on the fully coupled radiation solution. Conversion
includes destruction/back-conversion and therefore can be negative.
"""
    j, alpha, rho, dl = (np.asarray(x, dtype=float) for x in (j, alpha, rho, dl))
    n = len(dl)
    if substeps < 1 or j.shape != (n,4) or alpha.shape != (n,4) or rho.shape != (n,3):
        raise ValueError('inconsistent coefficient shapes/substeps')
    if any(not np.isfinite(x).all() for x in (j, alpha, rho, dl)) or np.any(dl < 0):
        raise ValueError('finite coefficients and nonnegative intervals required')
    if np.any(alpha[:,0] < 0):
        raise ValueError('CP attenuation budget currently requires nonnegative alpha_I')
    s = np.zeros(4) if initial is None else np.asarray(initial, dtype=float).copy()
    history = np.empty((n,4)); generation = np.zeros((n,3))
    channels = np.array([0.,0.,0.,s[3]])
    rates = rho.copy(); rates[:,:2] *= conversion_scale; rates[:,2] *= rotation_scale
    if split_index is not None:
        # Interventions only after the specified archived pre-transfer state.
        rates[:split_index] = rho[:split_index]
    for k in range(n):
        history[k] = s
        h = dl[k] / substeps
        prop = absorption_propagator(j[k], alpha[k], h)
        attenuation = np.exp(-alpha[k,0] * h)
        source_factor = -np.expm1(-alpha[k,0] * h) / alpha[k,0] if alpha[k,0] != 0 else h
        for _ in range(substeps):
            before = s
            half = rotation_step(before, rates[k], h/2)
            absorbed = (prop @ np.append(half, 1.))[:4]
            s = rotation_step(absorbed, rates[k], h/2)
            local = np.array([
                source_factor * j[k,3],
                absorbed[3] - attenuation * half[3] - source_factor * j[k,3],
                attenuation * (half[3] - before[3]) + s[3] - absorbed[3],
            ])
            channels *= attenuation; channels[:3] += local
            generation[k] = generation[k] * attenuation + local
    tau = alpha[:,0] * dl
    future_tau = np.cumsum(tau[::-1])[::-1] - tau
    observed_generation = generation * np.exp(-future_tau[:,None])
    return {'stokes':s, 'history':history, 'channels':channels,
            'observed_generation':observed_generation}


def quantiles(radius, weight):
    total = float(np.sum(weight))
    if total == 0: return None
    order = np.argsort(radius)
    return radius[order][np.searchsorted(np.cumsum(weight[order]), np.array([.05,.5,.95])*total)].tolist()


def trace_arrays(path, ray=0, freq_index=0):
    with h5py.File(path) as h:
        rays = h['rays']; n = int(rays['sample_count'][ray])
        multi = int(h.attrs.get('nfreq', 1)) > 1
        if freq_index < 0 or (not multi and freq_index != 0):
            raise ValueError('frequency index not available in this trace')
        group = h[f'trace/freq_{freq_index}'] if multi else h['trace']
        derived = group['derived'] if 'derived' in group else h['derived']
        if int(derived['complete'][ray]) != 1 or int(rays['reason'][ray]) != 1:
            raise ValueError('returned, complete unit-stride trace required')
        if n != int(rays['pass_b_steps'][ray]): raise ValueError('incomplete trace samples')
        def read(name):
            if name in group: data = group[name]
            elif f'trace/shared/{name}' in h: data = h[f'trace/shared/{name}']
            else: raise ValueError(f'missing trace field {name}')
            if data.dtype.itemsize != 8: raise ValueError('double-precision trace required for independent replay')
            if data.ndim == 2: return np.asarray(data[ray,:n])
            offset = int(rays['sample_offset'][ray])
            return np.asarray(data[offset:offset+n])
        recorded = np.stack([read('S'+s) for s in 'IQUV'], axis=1)
        # Legacy multi-frequency files only stored the first frequency at root.
        if 'rays' in group:
            terminal = group['rays']
        elif freq_index == 0:
            terminal = rays
        else:
            raise ValueError('selected frequency has no terminal Stokes; regenerate the trace')
        final = np.array([terminal[f'final_propagated_{s}_inv'][ray] for s in 'IQUV'])
        scale = float(final[0])
        if scale <= 0: raise ValueError('positive final intensity required')
        arrays = {
            'j': np.stack([read('j'+s) for s in 'IQUV'], axis=1)/scale,
            'alpha': np.stack([read('a'+s) for s in 'IQUV'], axis=1),
            'rho': np.stack([read('rho'+s) for s in 'QUV'], axis=1),
            'dl': np.abs(read('dlambda')),
        }
        radius = read('r'); lam = read('lambda')
        freeze = None
        if int(derived['intensity_freeze_valid'][ray]) == 1:
            freeze = int(np.argmin(abs(lam - derived['intensity_freeze_lambda'][ray])))
        metadata = {'input':str(Path(path).resolve()),'input_sha256':sha256(path),
                    'ray':ray,'freq_index':freq_index,'samples':n,
                    'frequency_hz':float(group.attrs.get('frequency_hz',h.attrs.get('frequency_hz',0))),
                    'intensity_freeze_index':freeze}
        return arrays, recorded/scale, final/scale, radius, metadata


def analyze_trace(path, ray=0, freq_index=0, closure_limit=1e-9):
    arrays, recorded, final, radius, metadata = trace_arrays(path, ray, freq_index)
    baseline = replay(**arrays, initial=recorded[0])
    closure = float(np.sum(abs(baseline['stokes']-final))/np.sum(abs(final)))
    history_error = float(np.max(abs(baseline['history']-recorded)))
    if max(closure,history_error) > closure_limit:
        raise ValueError(f'independent trace replay does not close: final={closure}, history={history_error}')
    def metrics(result):
        s=result['stokes']; p=complex(s[1],s[2]); pref=complex(final[1],final[2])
        return {'I_over_baseline_final_I':float(s[0]), 'linear_fraction':float(abs(p)/s[0]),
                'circular_fraction':float(s[3]/s[0]),
                'delta_evpa_deg':float(np.angle(p/pref)*90/np.pi) if abs(p)*abs(pref)>1e-20 else None,
                'stokes_over_baseline_final_I':s.tolist(),
                'CP_channels_over_baseline_final_I':dict(zip(CHANNELS,result['channels'].tolist())),
                'CP_channel_closure_over_baseline_final_I':float(sum(result['channels'])-s[3])}
    controls={'full':metrics(baseline)}
    settings={'no_rotation':(0.,1.,None),'no_conversion':(1.,0.,None),'no_faraday':(0.,0.,None)}
    freeze = metadata['intensity_freeze_index']
    if freeze is not None:
        settings.update({'no_rotation_after_I_freeze':(0.,1.,freeze),
                         'no_conversion_after_I_freeze':(1.,0.,freeze)})
    for name,(rv,rc,split) in settings.items():
        controls[name]=metrics(replay(**arrays, initial=recorded[0], rotation_scale=rv, conversion_scale=rc, split_index=split))
    refined={}
    for factor in (2,4):
        result=replay(**arrays, initial=recorded[0], substeps=factor)
        refined[str(factor)]=metrics(result)
        refined[str(factor)]['conversion_production_radius_quantiles_abs_weighted_M']=quantiles(radius,abs(result['observed_generation'][:,2]))
    refined_controls={}
    for name in ('no_rotation','no_conversion','no_conversion_after_I_freeze'):
        if name in settings:
            rv,rc,split=settings[name]
            refined_controls[name]=metrics(replay(**arrays, initial=recorded[0], substeps=4, rotation_scale=rv, conversion_scale=rc, split_index=split))
    production=baseline['observed_generation']
    abs_cp=np.abs(production[:,2])
    metadata.update({'independent_final_relative_l1':closure,
                     'independent_history_max_abs_over_final_I':history_error,
                     'controls':controls, 'fixed_coefficient_substep_refinement':refined,
                     'controls_at_four_substeps':refined_controls,
                     'conversion_production_radius_quantiles_abs_weighted_M':quantiles(radius,abs_cp),
                     'intrinsic_CP_emission_radius_quantiles_abs_weighted_M':quantiles(radius,abs(production[:,0])),
                     'conversion_post_freeze_abs_weight_fraction':float(abs_cp[freeze:].sum()/abs_cp.sum()) if freeze is not None and abs_cp.sum()>0 else None,
                     'conversion_absolute_integral_over_final_I':float(abs_cp.sum()),
                     'conversion_signed_integral_over_final_I':float(production[:,2].sum())})
    return metadata


def plot_mechanisms(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8, 'axes.titlesize':9})
    fig, axes=plt.subplots(1,2,figsize=(7.2,3.15),constrained_layout=True)
    labels=['Ray A', 'Ray B', 'Ray C']
    colors=['#426f9c','#d29346','#599a70','#333333']
    for k,(key,label) in enumerate(zip(CHANNELS[:3]+('total',),('Direct emission','Selective absorption','Conversion','Final V'))):
        values=[100*(r['controls']['full']['circular_fraction'] if key=='total' else r['controls']['full']['CP_channels_over_baseline_final_I'][key]) for r in rows]
        axes[0].bar(np.arange(len(rows))+(k-1.5)*.19,values,width=.18,color=colors[k],label=label)
    axes[0].axhline(0,color='black',lw=.5)
    axes[0].set_xticks(np.arange(len(rows)),labels)
    axes[0].set_ylabel(r'Signed contribution / final I [%]')
    axes[0].set_title('What sets the circular-polarization sign?')
    axes[0].legend(frameon=False,fontsize=7,loc='lower left')
    for k,(key,label,color) in enumerate((
        ('intrinsic_CP_emission_radius_quantiles_abs_weighted_M','Direct emission','#426f9c'),
        ('conversion_production_radius_quantiles_abs_weighted_M','Conversion','#599a70'))):
        q=np.array([r[key] for r in rows])
        axes[1].errorbar(q[:,1],np.arange(len(rows))+(k-.5)*.18,xerr=np.array([q[:,1]-q[:,0],q[:,2]-q[:,1]]),fmt='o',ms=4,capsize=3,color=color,label=label)
    axes[1].set_yticks(np.arange(len(rows)),labels); axes[1].invert_yaxis()
    axes[1].set_xscale('log'); axes[1].set_xlabel(r'Production radius [M]')
    axes[1].set_title('Where do the signed V terms arise?')
    axes[1].legend(frameon=False,fontsize=7)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces',nargs='+',type=Path)
    parser.add_argument('--ray',type=int,default=0)
    parser.add_argument('--freq-index',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args()
    rows=[]
    for path in args.traces:
        row=analyze_trace(path,args.ray,args.freq_index); rows.append(row)
        print(path, row['independent_final_relative_l1'], flush=True)
    result={'schema':'kpolaris_trace_mechanisms','schema_version':1,'script_sha256':sha256(__file__),
            'interpretation':'Signed CP production/destroying terms on the coupled radiation solution, propagated with scalar alpha_I attenuation. The sum closes to final V. These are production-site labels, distinct from seed-emission labels.',
            'interventions':'Prescribed plasma, emission, geometry and sample coefficients; change rho_V or (rho_Q,rho_U). Results include coupled feedback and cannot be added as independent effects.',
            'basis':'Parallel-transported path basis; polarization fractions and relative EVPA shifts. No absolute sky EVPA claim.',
            'refinement':'2/4 Strang subdivisions per frozen coefficient sample test splitting error, not interpolation/geodesic/fluid convergence.',
            'traces':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.figure: plot_mechanisms(rows,args.figure)


if __name__=='__main__': main()
