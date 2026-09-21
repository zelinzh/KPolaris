#!/usr/bin/env python3
"""End-to-end source/response validation against independent parameter reruns."""
import argparse
import json
import math
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_physical_responses import load_product, summarize, observable_response, observable_response_maps
from test_hdf5_contract import (write_synthetic_iharm_fixture, write_synthetic_athenak_fixture,
    synthetic_iharm_transport_args, synthetic_athenak_transport_args)


def read_stokes(path, freq=0):
    with h5py.File(path) as h:
        return np.stack([h[f'frame_0/freq_{freq}/{s}_inv'][...] for s in 'IQUV'])


def run_case(exe, base, work, name, extra, archive=False, timeout=None):
    output=work/(name+'.h5')
    args=[str(exe),*base,*extra,'--format=hdf5',f'--parameter_output={"auto" if archive else "none"}',f'--output={output}']
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
    (work/(name+'.log')).write_text(result.stdout+result.stderr)
    (work/(name+'.command.json')).write_text(json.dumps(args,indent=2)+'\n')
    if result.returncode:
        raise AssertionError(f'{name} failed: {result.stdout[-1000:]} {result.stderr[-2000:]}')
    return output


def validate(path,freq=0):
    row=summarize(*load_product(path,freq_index=freq),response_limit=1e-2)
    assert row['validation_passed'],row.get('validation')
    return row


def compare(a,b,tol=2e-9):
    err=float(np.abs(a-b).sum()/max(np.abs(b).sum(),np.finfo(float).tiny))
    assert err<tol,(err,tol)
    return err


def check_model(name,exe,base,work,mass_option,mass_value,full=False):
    # A parameter-independent sample grid makes the external rerun an exact
    # check of parameter-family semantics, separate from transfer convergence.
    base=[*base,'--max_radiation_depth=0','--max_absorption_depth=0',
          '--max_faraday_depth=0','--max_radiation_step=0.5']
    plain=run_case(exe,base,work,name+'_plain',['--analysis_mode=1'])
    baseline=read_stokes(plain)
    rows=[]
    settings=[('density_scale','region'),('temperature_scale','radial'),('magnetic_scale','plasma_region'),('coefficients','near_far')]
    if full:
        settings += [('none',x) for x in ('thetae','sigma','beta','ne_cgs','b_cgs')]
    density=None
    temperature=None
    for parameter,partition in settings:
        path=run_case(exe,base,work,f'{name}_{parameter}_{partition}',
            ['--analysis_mode=1',f'--analysis_response={parameter}',f'--analysis_partition={partition}'])
        row=validate(path); row['case']=path.stem
        compare(read_stokes(path),baseline,1e-14)
        rows.append(row)
        if parameter=='density_scale': density=path
        if parameter=='temperature_scale': temperature=path
    for sign,key in ((1,'plus_h'),(-1,'minus_h')):
        control=run_case(exe,base,work,f'{name}_external_{key}',
            ['--analysis_mode=1',f'--{mass_option}={mass_value*math.exp(sign*.001):.17g}'])
        predicted=load_product(density)[3][0 if sign>0 else 1]
        # AthenaK reconstructs its derived CGS fields into float32 storage on
        # each reload; scaling the already sampled baseline avoids re-quantization.
        tolerance=5e-7 if name=='athenak' else 2e-9
        rows[0][f'external_{key}_relative_l1']=compare(predicted,read_stokes(control),tolerance)
    if name=='riaf':
        for sign,key in ((1,'plus_h'),(-1,'minus_h')):
            control=run_case(exe,base,work,f'{name}_temperature_external_{key}',
                ['--analysis_mode=1',f'--riaf_Te0={math.exp(sign*.001):.17g}'])
            predicted=load_product(temperature)[3][0 if sign>0 else 1]
            rows[1][f'external_{key}_relative_l1']=compare(predicted,read_stokes(control))
    if full:
        # Both source maps and mechanism maps must be additive under merging bins.
        path=run_case(exe,base,work,name+'_one_bin',
            ['--analysis_mode=1','--analysis_response=density_scale','--analysis_partition=radial','--analysis_radial_bins=1'])
        a=load_product(density); b=load_product(path)
        compare(a[2].sum(axis=2),b[2].sum(axis=2),1e-11)
        rows.append(dict(case=path.stem,**validate(path)))
        archived=run_case(exe,base,work,name+'_custom_edges',
            ['--analysis_mode=1','--analysis_response=temperature_scale','--analysis_partition=thetae',
             '--analysis_partition_edges=2,5,20,80'],archive=True)
        repeated=run_case(exe,[],work,name+'_custom_roundtrip',
            ['--parameter_file='+str(archived)+'.params'])
        a=load_product(archived);b=load_product(repeated)
        compare(a[1],b[1],1e-14);compare(a[2],b[2],1e-14)
        rows.append(dict(case=repeated.stem,**validate(repeated)))
    return rows,baseline


def test_observable_derivatives():
    s=np.array([3.,.3,-.4,.02]); d=np.array([.2,.1,.04,-.03]); h=1e-5
    r=observable_response(s,d)
    def observ(x): return np.array([np.hypot(x[1],x[2])/x[0],x[3]/x[0],.5*np.arctan2(x[2],x[1])])
    fd=(observ(s+h*d)-observ(s-h*d))/(2*h)
    np.testing.assert_allclose(fd,[r['d_linear_fraction_dlogp'],r['d_circular_fraction_dlogp'],r['d_evpa_rad_dlogp']],rtol=1e-9)
    assert observable_response([1,0,0,0],[0,1,0,0])['d_evpa_rad_dlogp'] is None
    maps=observable_response_maps(s[:,None,None],d[:,None,None,None])
    np.testing.assert_allclose(maps['d_evpa_rad_dlogp'],r['d_evpa_rad_dlogp'])
    dark=observable_response_maps(np.zeros((4,1,1)),np.ones((4,2,1,1)))
    assert not dark['intensity_valid'].any()
    assert np.isnan(dark['d_evpa_rad_dlogp']).all()
    baseline=np.array([1.,.2,0,0])[:,None,None]
    tags=np.stack([baseline*.4,baseline*.6],axis=1)
    meta={'response_available':False}
    assert summarize(baseline,tags,None,None,meta)['validation_passed']
    bad=tags.copy();bad[0,0]*=.5
    try: summarize(baseline,bad,None,None,meta)
    except ValueError: pass
    else: raise AssertionError('broken source closure was accepted')
    meta.update(response_available=True,log_parameter_step=.001)
    bad_response=np.ones((4,4,2,1,1))
    reruns=np.repeat(baseline[None],4,axis=0)
    assert not summarize(baseline,tags,bad_response,reruns,meta)['validation_passed']


def test_source_closure_underflow():
    # A bright pixel keeps the observable finite; the dark pixel has a
    # one-ULP discrepancy at the double underflow limit, as in a RIAF image.
    stokes=np.zeros((4,1,2),dtype=np.float64)
    stokes[0,0]=[1.,1e-320]
    tags=stokes[:,None].copy()
    tags[0,0,0,1]=np.nextafter(stokes[0,0,1],np.inf)
    result=summarize(stokes,tags,None,None,{'response_available':False})
    assert result['validation_passed']
    assert result['source_closure_subnormal_pixels']==1
    assert result['source_closure_normalization_floor']==np.finfo(float).tiny
    for value in (1.,1e-100,1e-300):
        stokes[0,0,1]=value
        tags[0,0,0,1]=.99*value
        try: summarize(stokes,tags,None,None,{'response_available':False})
        except ValueError as error: assert 'do not close' in str(error)
        else: raise AssertionError('normal-range closure error was accepted')
    tags[0,0,0,1]=0.
    try: summarize(stokes,tags,None,None,{'response_available':False})
    except ValueError as error: assert 'do not close' in str(error)
    else: raise AssertionError('missing source contribution was accepted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--riaf',type=Path)
    parser.add_argument('--iharm',type=Path)
    parser.add_argument('--athenak',type=Path)
    parser.add_argument('--workdir',type=Path)
    args=parser.parse_args()
    test_observable_derivatives()
    test_source_closure_underflow()
    temporary=tempfile.TemporaryDirectory(prefix='kpolaris-response-') if args.workdir is None else None
    work=(args.workdir or Path(temporary.name)).resolve(); work.mkdir(parents=True,exist_ok=True)
    rows=[]
    if args.riaf:
        base=['--parameter_file='+str(ROOT/'params/riaf_recommended.par'),'--nx=8','--ny=8']
        result,_=check_model('riaf',args.riaf.resolve(),base,work,'riaf_ne_unit',5e6,True);rows+=result
    for name,exe,writer,argmaker in [('iharm',args.iharm,write_synthetic_iharm_fixture,synthetic_iharm_transport_args),
                                    ('athenak',args.athenak,write_synthetic_athenak_fixture,synthetic_athenak_transport_args)]:
        if not exe: continue
        fixture=work/(name+('.h5' if name=='iharm' else '.bin'))
        writer(fixture);base=argmaker(fixture)
        result,baseline=check_model(name,exe.resolve(),base,work,name+'_M_unit',3e25); rows+=result
        # Repeated static states exercise temporal-window handoff and final
        # observer-basis conversion. The second frequency has its own buffers.
        slow=run_case(exe.resolve(),base,work,name+'_slow_multifrequency',
            ['--analysis_mode=1','--analysis_response=temperature_scale','--analysis_partition=thetae',
             '--slow_light=1','--slow_light_observation_time=200',
             f'--slow_light_dump_list={fixture},{fixture},{fixture}',
             '--slow_light_time_list=0,100,1000','--freq_list=230000000000,345000000000'])
        for f in (0,1): rows.append(dict(case=slow.stem,freq_index=f,**validate(slow,f)))
        # Compare complete slow-light products with fast-light at the same
        # transfer limits; windows may introduce additional sampling points.
        fast=run_case(exe.resolve(),base,work,name+'_fast_matched',
            ['--analysis_mode=1','--analysis_response=temperature_scale','--analysis_partition=thetae',
             '--freq_list=230000000000,345000000000'])
        for f in (0,1):
            a=load_product(slow,freq_index=f);b=load_product(fast,freq_index=f)
            rows[-2+f]['static_slow_fast_stokes_relative_l1']=compare(a[0],b[0],.02)
            rows[-2+f]['static_slow_fast_derivative_relative_l1']=compare(a[2],b[2],.03)
        # Non-identical snapshots exercise perturb-before-time-interpolation
        # and changing source-bin membership, rather than only the static limit.
        changed=work/(name+'_changed'+fixture.suffix)
        shutil.copyfile(fixture,changed)
        if name=='iharm':
            with h5py.File(changed,'r+') as h:
                prim=h['prims'][...]; prim[...,1]*=1.5; prim[...,5:8]*=.8; h['prims'][...]=prim
        else:
            raw=bytearray(changed.read_bytes())
            prim=np.frombuffer(raw,dtype='<f4',offset=len(raw)-8*8*8*8*4).reshape(8,8,8,8)
            prim[4]*=1.5; prim[5:8]*=.8; changed.write_bytes(raw)
        evolving=run_case(exe.resolve(),base,work,name+'_evolving_slow',
            ['--analysis_mode=1','--analysis_response=magnetic_scale','--analysis_partition=thetae',
             '--slow_light=1','--slow_light_observation_time=200',
             f'--slow_light_dump_list={fixture},{changed},{fixture}',
             '--slow_light_time_list=0,100,1000','--freq_list=230000000000,345000000000'])
        for f in (0,1): rows.append(dict(case=evolving.stem,freq_index=f,**validate(evolving,f)))
    summary={'cases':len(rows),'all_passed':True,'results':rows}
    (work/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'Physical response integration tests passed: {len(rows)} cases; {work}')


if __name__=='__main__': main()
