#!/usr/bin/env python3
"""Validate and summarize full-Stokes physical responses and source partitions.

Reads native analysis/physical_response products. Derivatives are with respect
to ln(parameter scale); mechanism responses and source tags retain their signs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

STOKES = 'IQUV'
MECHANISMS = ('emission', 'absorption', 'rotation', 'conversion')


def string(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def observable_response(stokes, derivative, polarization_floor=1e-10):
    """Absolute and fractional responses of image-integrated observables."""
    i, q, u, v = map(float, stokes)
    di, dq, du, dv = map(float, derivative)
    if not i > 0:
        raise ValueError('positive baseline intensity required')
    p = complex(q, u)
    dp = complex(dq, du)
    out = {'d_stokes_over_baseline_I_dlogp': [di/i, dq/i, du/i, dv/i],
           'dln_flux_I_dlogp': di/i,
           'd_circular_fraction_dlogp': (dv-v*di/i)/i,
           'linear_response_valid': abs(p) > polarization_floor*i}
    if out['linear_response_valid']:
        z = dp/p
        out.update(dln_linear_amplitude_dlogp=float(z.real),
                   d_linear_fraction_dlogp=float(abs(p)/i*(z.real-di/i)),
                   d_evpa_rad_dlogp=float(.5*z.imag))
    else:
        out.update(dln_linear_amplitude_dlogp=None,
                   d_linear_fraction_dlogp=None, d_evpa_rad_dlogp=None)
    return out


def load_product(path, frame=0, freq_index=0):
    with h5py.File(path) as h:
        fg = h[f'frame_{frame}']
        image = fg[f'freq_{freq_index}']
        if 'analysis' in image:
            analysis = image['analysis']
        elif int(h.attrs.get('nfreq', 1)) == 1 and freq_index == 0:
            analysis = fg['analysis']
        else:
            raise ValueError('frequency has no independent analysis group')
        g = analysis['physical_response']
        if int(g.attrs['schema_version']) != 1:
            raise ValueError('unsupported physical response schema')
        reason = np.asarray(fg['diagnostics/reason'])
        if np.any(reason != 1):
            raise ValueError(f'{np.count_nonzero(reason != 1)} rays did not return; complete images required')
        stokes = np.stack([image[s+'_inv'][...] for s in STOKES])
        tags = np.stack([g['source/'+s+'_inv'][...] for s in STOKES])
        bins = int(g.attrs['bins'])
        if stokes.shape != (4,)+reason.shape or tags.shape != (4,bins)+reason.shape:
            raise ValueError('source/image shape mismatch')
        metadata = {k: string(g.attrs[k]) for k in ('partition', 'partition_semantics', 'source_semantics')}
        metadata['bins'] = bins
        metadata['emission_selection'] = {
            'equatorial_h_over_r': float(h.attrs.get('equatorial_h_over_r', 0)),
            'equatorial_samples': int(h.attrs.get('equatorial_samples', 8)),
            'faraday_rotation': int(h.attrs.get('faraday_rotation', 1)),
            'direct_only': int(h.attrs.get('direct_only', 0))}
        for key in ('equatorial_emission_definition', 'equatorial_transfer_definition'):
            if key in h.attrs: metadata['emission_selection'][key] = string(h.attrs[key])
        if 'bin_labels' in g.attrs:
            metadata['bin_labels'] = string(g.attrs['bin_labels']).split(',')
        if 'partition_edges' in g:
            metadata['partition_edges'] = g['partition_edges'][...].tolist()
        metadata['response_available'] = bool(g.attrs['response_available'])
        derivative = reruns = None
        if metadata['response_available']:
            derivative = np.stack([np.stack([g[f'derivative/{m}/d{s}_inv_dlogp'][...] for s in STOKES]) for m in MECHANISMS])
            reruns = np.stack([np.stack([g[f'reruns/{v}/{s}_inv'][...] for s in STOKES]) for v in ('plus_h','minus_h','plus_half_h','minus_half_h')])
            if derivative.shape != (4,4,bins)+reason.shape or reruns.shape != (4,4)+reason.shape:
                raise ValueError('response/image shape mismatch')
            metadata['parameter'] = string(g.attrs['parameter'])
            metadata['log_parameter_step'] = float(g.attrs['log_parameter_step'])
            if not np.isfinite(metadata['log_parameter_step']) or metadata['log_parameter_step'] <= 0:
                raise ValueError('invalid logarithmic parameter step')
            metadata['fixed_quantities'] = string(g.attrs['fixed_quantities'])
        for x in (stokes,tags,derivative,reruns):
            if x is not None and not np.isfinite(x).all():
                raise ValueError('nonfinite Stokes or response values')
    return stokes, tags, derivative, reruns, metadata


def summarize(stokes, tags, derivative, reruns, metadata,
              closure_limit=1e-10, response_limit=1e-2):
    if not np.isfinite([closure_limit,response_limit]).all() or min(closure_limit,response_limit)<=0:
        raise ValueError("validation limits must be finite and positive")
    result = dict(metadata)
    image_axes = tuple(range(1, stokes.ndim))
    total = stokes.sum(axis=image_axes)
    if total[0] <= 0:
        raise ValueError('positive total intensity required')
    # Normalize closure per pixel by tagged amplitude, so cancellation does
    # not hide missing components; an all-zero pixel has zero closure.
    residual = np.abs(tags.sum(axis=1)-stokes).sum(axis=0)
    scale = np.abs(tags).sum(axis=(0,1))
    # Subnormal numbers do not retain relative double precision. A single ULP
    # around 1e-320 must not become a macroscopic closure failure. Use the
    # smallest normal value only as the normalization floor; keep the image,
    # tags and the relative tolerance unchanged in the normal range.
    normalization_floor = np.finfo(scale.dtype).tiny
    closure = residual / np.maximum(scale, normalization_floor)
    closure[(scale==0)&(residual>0)] = np.inf
    result['source_closure_max_relative_l1'] = float(closure.max())
    result['source_closure_normalization_floor'] = float(normalization_floor)
    result['source_closure_subnormal_pixels'] = int(np.count_nonzero((scale>0)&(scale<normalization_floor)))
    if result['source_closure_max_relative_l1'] > closure_limit:
        raise ValueError('source tags do not close against observed Stokes')
    result['integrated_stokes_over_I'] = (total/total[0]).tolist()
    integrated_tags = tags.sum(axis=tuple(range(2,tags.ndim)))
    result['source_bins'] = [
        {'bin': k, 'stokes_over_total_I': (integrated_tags[:,k]/total[0]).tolist(),
         'resolved_linear_amplitude_over_total_I': float(np.hypot(tags[1,k],tags[2,k]).sum()/total[0]),
         'resolved_circular_amplitude_over_total_I': float(np.abs(tags[3,k]).sum()/total[0])}
        for k in range(tags.shape[1])]
    if derivative is None:
        result['validation_passed'] = True
        return result
    h = result['log_parameter_step']
    summed = derivative.sum(axis=(0,2))
    fd = (reruns[0]-reruns[1])/(2*h)
    fd_half = (reruns[2]-reruns[3])/h
    # Absolute error in units of baseline I remains useful if a derivative is zero.
    err = float(np.abs(summed-fd_half).sum())
    refinement = float(np.abs(fd-fd_half).sum())
    norm = float(np.abs(fd_half).sum())
    denom = max(norm, 1e-10*float(total[0]))
    result['validation'] = {
        'decomposition_vs_full_half_step_relative_l1': err/denom,
        'full_difference_step_refinement_relative_l1': refinement/denom,
        'decomposition_error_over_total_I': err/float(total[0]),
        'step_refinement_error_over_total_I': refinement/float(total[0]),
        'response_limit': response_limit,
    }
    result['validation_passed'] = max(err,refinement)/denom <= response_limit
    result['total_response'] = observable_response(total,summed.sum(axis=image_axes))
    result['mechanisms'] = {}
    for m,name in enumerate(MECHANISMS):
        summed_bins = derivative[m].sum(axis=1)
        result['mechanisms'][name] = {
            'integrated': observable_response(total,summed_bins.sum(axis=image_axes)),
            'bins': [dict(bin=k, **observable_response(total,derivative[m,:,k].sum(axis=image_axes)))
                     for k in range(tags.shape[1])],
        }
    return result



def observable_response_maps(stokes, derivative, intensity_floor=1e-10, polarization_floor=1e-10):
    """Pixel responses, broadcasting over optional leading bin dimensions.

    derivative has shape (4, ..., ny, nx). Undefined ratios are NaN with
    explicit masks; absolute Stokes derivatives remain available everywhere.
    """
    i,q,u,v=stokes
    di,dq,du,dv=derivative
    intensity_valid=i>intensity_floor*max(float(i.max()),np.finfo(float).tiny)
    p2=q*q+u*u
    linear_valid=intensity_valid & (p2>(polarization_floor*i)**2)
    def ratio(numerator,denominator,valid):
        result=np.full(np.broadcast_shapes(numerator.shape,denominator.shape),np.nan)
        return np.divide(numerator,denominator,out=result,where=valid)
    dln_i=ratio(di,i,intensity_valid)
    dln_p=ratio(q*dq+u*du,p2,linear_valid)
    return {
        'intensity_valid':intensity_valid.astype(np.uint8),
        'linear_valid':linear_valid.astype(np.uint8),
        'dln_I_dlogp':dln_i,
        'd_linear_fraction_dlogp':ratio(np.sqrt(p2),i,linear_valid)*(dln_p-dln_i),
        'd_circular_fraction_dlogp':ratio(dv-v*dln_i,i,intensity_valid),
        'd_evpa_rad_dlogp':ratio(q*du-u*dq,2*p2,linear_valid),
    }


def write_observable_maps(stokes,derivative,metadata,path,validation=None):
    if derivative is None:
        raise ValueError('--maps requires a product with responses')
    with h5py.File(path,'w') as h:
        h.attrs['schema']='kpolaris_observable_response_v1'
        if validation is not None:
            h.attrs['validation_passed']=int(validation['validation_passed'])
            h.attrs['validation_json']=json.dumps(validation.get('validation',{}),allow_nan=False)
        h.attrs['parameter']=metadata['parameter']
        h.attrs['partition']=metadata['partition']
        for key,value in metadata.get('emission_selection',{}).items():
            h.attrs[key]=value
        h.attrs['derivative_coordinate']='ln(parameter scale)'
        h.attrs['intensity_floor_relative_peak']=1e-10
        h.attrs['polarization_floor_relative_local_I']=1e-10
        h.attrs['undefined_ratio']='NaN; consult intensity_valid and linear_valid masks'
        def write(name,d):
            g=h.create_group(name)
            for s,array in zip(STOKES,d): g.create_dataset('d'+s+'_inv_dlogp',data=array,compression='gzip')
            for key,array in observable_response_maps(stokes,d).items():
                g.create_dataset(key,data=array,compression='gzip')
        write('total',derivative.sum(axis=(0,2)))
        for m,name in enumerate(MECHANISMS):
            write('mechanisms/'+name,derivative[m].sum(axis=1))
            write('partition/'+name,derivative[m])


def plot_product(stokes, tags, derivative, metadata, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    if derivative is None:
        n = min(tags.shape[1],8)
        fig,axes=plt.subplots(1,n,figsize=(2.2*n,2.6),squeeze=False,layout='constrained')
        peak=max(float(stokes[0].max()),np.finfo(float).tiny)
        labels=metadata.get('bin_labels',[f'Bin {k}' for k in range(tags.shape[1])])
        for k,ax in enumerate(axes[0]):
            im=ax.imshow(tags[0,k]/peak,origin='lower',cmap='magma',vmin=0,vmax=1)
            ax.set_title(labels[k],fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Tagged I / peak total I',shrink=.7)
    else:
        maps=derivative.sum(axis=2)/max(float(stokes[0].max()),np.finfo(float).tiny)
        fig,axes=plt.subplots(4,4,figsize=(9,8),layout='constrained')
        for s in range(4):
            vmax=max(float(np.abs(maps[:,s]).max()),1e-12)
            norm=SymLogNorm(linthresh=vmax*.01,vmin=-vmax,vmax=vmax)
            for m in range(4):
                im=axes[m,s].imshow(maps[m,s],origin='lower',cmap='RdBu_r',norm=norm)
                axes[m,s].set_xticks([]); axes[m,s].set_yticks([])
                if m==0: axes[m,s].set_title(f'd{STOKES[s]} / d ln p')
                if s==0: axes[m,s].set_ylabel(MECHANISMS[m].capitalize())
            fig.colorbar(im,ax=axes[:,s].tolist(),shrink=.65)
        fig.suptitle(metadata['parameter']+'; response / peak baseline I')
    fig.savefig(output,dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--freq-index',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True,help='JSON summary')
    parser.add_argument('--plot',type=Path)
    parser.add_argument('--maps',type=Path,help='HDF5 pixel/partition responses of polarization fraction and EVPA')
    parser.add_argument('--closure-limit',type=float,default=1e-10)
    parser.add_argument('--response-limit',type=float,default=1e-2)
    args=parser.parse_args()
    if not np.isfinite([args.closure_limit,args.response_limit]).all() or args.closure_limit<=0 or args.response_limit<=0:
        parser.error('validation limits must be positive')
    arrays=load_product(args.image,args.frame,args.freq_index)
    result=summarize(*arrays,closure_limit=args.closure_limit,response_limit=args.response_limit)
    with args.image.open('rb') as stream:
        result.update(input=str(args.image.resolve()),input_sha256=hashlib.file_digest(stream,'sha256').hexdigest(),
                      frame=args.frame,freq_index=args.freq_index)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.maps:
        args.maps.parent.mkdir(parents=True,exist_ok=True)
        write_observable_maps(arrays[0],arrays[2],arrays[4],args.maps,result)
    if args.plot:
        args.plot.parent.mkdir(parents=True,exist_ok=True)
        plot_product(arrays[0],arrays[1],arrays[2],arrays[4],args.plot)
    print(json.dumps({'validation_passed':result['validation_passed'],
                      'source_closure':result['source_closure_max_relative_l1'],
                      'validation':result.get('validation')}))
    if not result['validation_passed']:
        raise SystemExit('Response validation failed: reduce analysis_response_step and verify transfer step convergence.')


if __name__=='__main__':
    main()
