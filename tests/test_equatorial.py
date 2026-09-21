#!/usr/bin/env python3
"""Native thin-equatorial emission: transfer, convergence, response and slow light."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import h5py
import numpy as np
from test_physical_response import run_case, read_stokes, compare, validate, load_product, ROOT
from analyze_physical_responses import write_observable_maps
from test_hdf5_contract import (write_synthetic_iharm_fixture, write_synthetic_athenak_fixture,
    synthetic_iharm_transport_args, synthetic_athenak_transport_args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for model in ('riaf','iharm','athenak'): parser.add_argument('--'+model,type=Path)
    parser.add_argument('--workdir',type=Path)
    args=parser.parse_args();temp=tempfile.TemporaryDirectory() if args.workdir is None else None
    work=(args.workdir or Path(temp.name)).resolve();work.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model in ('riaf','iharm','athenak'):
        exe=getattr(args,model)
        if exe is None: continue
        exe=exe.resolve();fixture=None
        if model=='riaf':
            base=['--parameter_file='+str(ROOT/'params/riaf_recommended.par'),'--nx=6','--ny=6',
                  '--radius=100','--inclination_deg=17','--xspan=10','--riaf_r_max=30']
        else:
            fixture=work/(model+('.h5' if model=='iharm' else '.bin'))
            (write_synthetic_iharm_fixture if model=='iharm' else write_synthetic_athenak_fixture)(fixture)
            base=(synthetic_iharm_transport_args if model=='iharm' else synthetic_athenak_transport_args)(fixture)
            base+=['--nx=6','--ny=6','--xspan=10','--fovx_dsource=80','--fovy_dsource=80']
        base+=['--adaptive_tolerance=1e-12','--max_step=.25','--max_radiation_step=.05']
        def run(label,extra=(),archive=False):
            path=run_case(exe,base,work,model+'_'+label,list(extra),archive,timeout=180)
            with h5py.File(path) as h:
                assert np.all(h['frame_0/diagnostics/reason'][...]==1),label
                for key in ('equatorial_h_over_r','equatorial_samples','faraday_rotation'):
                    assert key in h.attrs and key in h['parameters/radiation'].attrs,key
            return path
        full=run('full')
        for label,opts in [('off',['--equatorial_h_over_r=0']),('sphere',['--equatorial_h_over_r=1'])]:
            assert np.array_equal(read_stokes(full),read_stokes(run(label,opts)))
        wedge=['--equatorial_h_over_r=.01']
        thin=run('thin',wedge,True);thin_s=read_stokes(thin);full_s=read_stokes(full)
        fraction=float(thin_s[0].sum()/full_s[0].sum());assert 0<fraction<.9,fraction
        with h5py.File(full) as h0,h5py.File(thin) as h1:
            assert np.array_equal(h0['frame_0/diagnostics/pass_a_steps'][...],h1['frame_0/diagnostics/pass_a_steps'][...]),'emission mask changes geodesics'
            assert '2606.12518' in str(h1.attrs['equatorial_reference'])
        row={'model':model,'thin_flux_fraction':fraction}
        fused=run('fused',wedge+['--split_transport=0']);row['split_fused_l1']=compare(read_stokes(fused),thin_s,1e-6)
        replay=run('replay',['--parameter_file='+str(thin)+'.params']);compare(read_stokes(replay),thin_s,1e-12)
        refined=run('refined',wedge+['--equatorial_samples=16','--max_radiation_step=.025'])
        row['sampling_refinement_l1']=compare(read_stokes(refined),thin_s,.015)
        rotation_off=run('no_rotation',wedge+['--faraday_rotation=0','--analysis_mode=1'])
        with h5py.File(rotation_off) as h:
            assert np.count_nonzero(h['frame_0/analysis/faraday_rotation_depth'][...])==0
            assert np.count_nonzero(h['frame_0/analysis/faraday_conversion_depth'][...])>0
        response=run('response',wedge+['--faraday_rotation=0','--analysis_mode=1',
                     '--analysis_response=density_scale','--analysis_partition=region'])
        row['response_validation']=validate(response)['validation']
        product=load_product(response)
        assert np.count_nonzero(product[2][2])==0,'rhoV=0 must also remove the rotation response'
        maps=work/(model+'_observable_maps.h5')
        write_observable_maps(product[0],product[2],product[4],maps)
        with h5py.File(maps) as h:
            assert h.attrs['equatorial_h_over_r']==.01 and h.attrs['faraday_rotation']==0
            assert h.attrs['equatorial_samples']==8 and h.attrs['direct_only']==0
        compare(read_stokes(response),read_stokes(rotation_off),1e-6)
        # An independent density rerun checks that source selection and rhoV=0
        # also apply to physically recomputed coefficients, not just base images.
        if model=='riaf':
            density=float(next(line.split('=',1)[1] for line in (Path(str(thin)+'.params')).read_text().splitlines() if line.startswith('riaf_ne_unit=')))
            ext=run('density_plus',wedge+['--faraday_rotation=0','--analysis_mode=1',f'--riaf_ne_unit={density*math.exp(.001):.17g}'])
            row['external_density_l1']=compare(read_stokes(ext),load_product(response)[3][0],2e-5)
        direct=run('direct',wedge+['--direct_only=1'])
        assert 0<float(read_stokes(direct)[0].sum())<=thin_s[0].sum()*1.005
        if fixture:
            freqs=['--freq_list=230000000000,345000000000']
            multi=run('multi',wedge+freqs)
            control=run('control',wedge+freqs+['--split_transport=0','--multifrequency_chunk_size=1'])
            fused_multi=run('fused_multi',wedge+freqs+['--split_transport=0'])
            for f in (0,1):
                compare(read_stokes(multi,f),read_stokes(control,f),1e-6)
                compare(read_stokes(multi,f),read_stokes(fused_multi,f),1e-6)
            slowargs=wedge+freqs+['--slow_light=1','--slow_light_observation_time=200',
                f'--slow_light_dump_list={fixture},{fixture},{fixture}','--slow_light_time_list=0,100,1000']
            slow=run('slow',slowargs)
            slowresponse=run('slow_response',slowargs+['--analysis_mode=1',
                '--analysis_response=temperature_scale','--analysis_partition=region'])
            row['slow_fast_l1']=[]
            for f in (0,1):
                row['slow_fast_l1'].append(compare(read_stokes(slow,f),read_stokes(multi,f),.005))
                validate(slowresponse,f)
                compare(read_stokes(slowresponse,f),read_stokes(slow,f),.005)
        if fixture:
            for analysis in (0,1):
                failed=run_case(exe,base,work,f'{model}_slow_underflow_{analysis}',slowargs+[
                    '--equatorial_h_over_r=1e-6','--adaptive=0','--min_step=.1',
                    '--step=.025',f'--analysis_mode={analysis}'],timeout=180)
                with h5py.File(failed) as h:
                    assert np.all(h['frame_0/diagnostics/reason'][...]==7)
        for option in ('equatorial_h_over_r=-.1','equatorial_h_over_r=nan','equatorial_h_over_r=1.1',
                       'equatorial_h_over_r=1e-9','equatorial_samples=1','equatorial_samples=8.5','faraday_rotation=2'):
            bad=subprocess.run([str(exe),*base,'--'+option],capture_output=True,text=True,timeout=30)
            assert bad.returncode and option.split('=')[0] in bad.stdout+bad.stderr,option
        # A deliberately incompatible minimum step must remain a failure in
        # plain, analysis, and slow-light finalization (reason 7, not success).
        if model=='riaf':
            for analysis in (0,1):
                failed=run_case(exe,base,work,f'{model}_underflow_{analysis}',
                    ['--equatorial_h_over_r=1e-6','--adaptive=0','--min_step=.1',
                     '--step=.025',f'--analysis_mode={analysis}'],timeout=180)
                with h5py.File(failed) as h:
                    assert np.all(h['frame_0/diagnostics/reason'][...]==7)
        rows.append(row)
    (work/'summary.json').write_text(json.dumps({'all_passed':True,'models':rows},indent=2,allow_nan=False)+'\n')
    print(f'Equatorial native tests passed: {len(rows)} models; {work}')

if __name__=='__main__':main()
