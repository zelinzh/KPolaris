#!/usr/bin/env python3
"""Native direct-only integration: split/fused, response tags, slow light and frequencies."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import numpy as np
import h5py
from test_physical_response import run_case, read_stokes, compare, validate, ROOT
from test_hdf5_contract import (write_synthetic_iharm_fixture,write_synthetic_athenak_fixture,
                                synthetic_iharm_transport_args,synthetic_athenak_transport_args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for model in ('riaf','iharm','athenak'):p.add_argument('--'+model,type=Path)
    p.add_argument('--workdir',type=Path)
    p.add_argument('--max-frequencies',type=int,default=2)
    a=p.parse_args(); temp=tempfile.TemporaryDirectory() if a.workdir is None else None
    work=(a.workdir or Path(temp.name)).resolve();work.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model in ('riaf','iharm','athenak'):
        exe=getattr(a,model)
        if exe is None:continue
        exe=exe.resolve();fixture=None
        if model=='riaf':
            base=['--parameter_file='+str(ROOT/'params/riaf_recommended.par'),'--nx=6','--ny=6','--radius=100','--inclination_deg=17','--xspan=10','--riaf_r_max=30']
        else:
            fixture=work/(model+('.h5' if model=='iharm' else '.bin'))
            (write_synthetic_iharm_fixture if model=='iharm' else write_synthetic_athenak_fixture)(fixture)
            base=(synthetic_iharm_transport_args if model=='iharm' else synthetic_athenak_transport_args)(fixture)
            base+=['--nx=6','--ny=6','--xspan=10','--fovx_dsource=80','--fovy_dsource=80','--max_radiation_step=.1','--max_step=.5','--adaptive_tolerance=1e-10']
        def run(label,extra,archive=False):
            path=run_case(exe,base,work,model+'_'+label,extra,archive,timeout=120)
            with h5py.File(path) as h:
                assert np.all(h['frame_0/diagnostics/reason'][...]==1),label
                direct=int(h.attrs['direct_only']); assert direct in (0,1)
                if direct:
                    assert '2512.09641' in str(h.attrs['direct_only_reference'])
                    assert 'first observer-to-source' in str(h.attrs['direct_only_boundary'])
            return path
        full=run('all',[])
        off=run('off',['--direct_only=0'])
        assert np.array_equal(read_stokes(full),read_stokes(off))
        direct=run('direct',['--direct_only=1','--split_transport=1'],True)
        fused=run('fused',['--direct_only=1','--split_transport=0'])
        row={'model':model,'split_fused_stokes_l1':compare(read_stokes(direct),read_stokes(fused),1e-6)}
        all_s,dir_s=read_stokes(full),read_stokes(direct)
        row['direct_flux_fraction']=float(dir_s[0].sum()/all_s[0].sum())
        with h5py.File(full) as h0,h5py.File(direct) as h1:
            row['shortened_rays']=int(np.count_nonzero(h1['frame_0/diagnostics/pass_a_steps'][...] < h0['frame_0/diagnostics/pass_a_steps'][...]))
        assert row['shortened_rays']>0,row
        assert 0<row['direct_flux_fraction']<=1.001,row
        assert np.max(np.abs(all_s-dir_s))>1e-8*all_s[0].max(),row
        assert np.min(all_s[0]-dir_s[0])>-1e-3*all_s[0].max()
        response=run('response',['--direct_only=1','--analysis_mode=1','--analysis_response=temperature_scale','--analysis_partition=region'])
        result=validate(response);row['response_closure']=result['source_closure_max_relative_l1'];row['response_fd']=result['validation']
        row['analysis_normal_stokes_l1']=compare(read_stokes(response),dir_s,1e-6)
        replay=run('replay',['--parameter_file='+str(direct)+'.params'])
        compare(read_stokes(replay),dir_s,1e-6)
        # Refining geometric steps should preserve the cutoff and converge the image.
        refined=run('refined',['--direct_only=1','--max_step=.25','--adaptive_tolerance=1e-12','--max_radiation_step=.05'])
        row['refined_image_stokes_l1']=compare(read_stokes(refined),dir_s,.03)
        if fixture:
            # Keep the slow-light checks in single-frequency builds as well.
            frequencies=['230000000000','345000000000'][:min(2,a.max_frequencies)]
            freq=['--freq_list='+','.join(frequencies)]
            fast=run('multi',['--direct_only=1',*freq])
            row['multi_single_stokes_l1']=compare(read_stokes(fast),dir_s,.003)
            control=run('control',['--direct_only=1',*freq,'--multifrequency_chunk_size=1','--split_transport=0'])
            fused_multi=run('fused_multi',['--direct_only=1',*freq,'--split_transport=0'])
            for f in range(len(frequencies)):
                compare(read_stokes(fast,f),read_stokes(control,f),1e-6)
                compare(read_stokes(fast,f),read_stokes(fused_multi,f),1e-6)
            slowargs=['--direct_only=1','--slow_light=1','--slow_light_observation_time=200',
                f'--slow_light_dump_list={fixture},{fixture},{fixture}','--slow_light_time_list=0,100,1000',*freq]
            slow=run('slow',slowargs)
            no_prefetch=run('slow_no_prefetch',slowargs+['--slow_light_prefetch=0'])
            full_slow=run('slow_all',slowargs+['--direct_only=0'])
            for f in range(len(frequencies)):
                compare(read_stokes(no_prefetch,f),read_stokes(slow,f),1e-6)
            changed=work/(model+'_changed'+fixture.suffix)
            shutil.copyfile(fixture,changed)
            if model=='iharm':
                with h5py.File(changed,'r+') as h:
                    prim=h['prims'][...];prim[...,1]*=1.5;prim[...,5:8]*=.8;h['prims'][...]=prim
            else:
                raw=bytearray(changed.read_bytes())
                prim=np.frombuffer(raw,dtype='<f4',offset=len(raw)-8*8*8*8*4).reshape(8,8,8,8)
                prim[4]*=1.5;prim[5:8]*=.8;changed.write_bytes(raw)
            evolving=run('slow_evolving',slowargs+[f'--slow_light_dump_list={fixture},{changed},{fixture}',
                '--analysis_mode=1','--analysis_response=magnetic_scale','--analysis_partition=region'])
            row['evolving_slow_responses']=[validate(evolving,f)['validation'] for f in range(len(frequencies))]
            row['evolving_static_difference_l1']=float(np.abs(read_stokes(evolving)-read_stokes(slow)).sum()/np.abs(read_stokes(slow)).sum())
            assert row['evolving_static_difference_l1']>1e-5
            slowresponse=run('slow_response',slowargs+['--analysis_mode=1','--analysis_response=temperature_scale','--analysis_partition=region'])
            row['slow_fast_stokes_l1']=[];row['slow_responses']=[]
            for f in range(len(frequencies)):
                row['slow_fast_stokes_l1'].append(compare(read_stokes(slow,f),read_stokes(fast,f),.003))
                result=validate(slowresponse,f);row['slow_responses'].append(result['validation'])
                compare(read_stokes(slowresponse,f),read_stokes(slow,f),.02)
        bad=subprocess.run([str(exe),*base,'--direct_only=2','--output='+str(work/'invalid.h5')],capture_output=True,text=True)
        assert bad.returncode and 'direct_only must be 0 or 1' in bad.stdout+bad.stderr
        rows.append(row)
    (work/'summary.json').write_text(json.dumps({'all_passed':True,'models':rows},indent=2,allow_nan=False)+'\n')
    print(f'Direct-only native tests passed: {len(rows)} models; {work}')

if __name__=='__main__':main()
