#!/usr/bin/env python3
"""Exercise temporal interpolation through frequency, analysis and trace drivers."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from test_hdf5_contract import (
    run, text_attr, synthetic_kharma_transport_args, write_synthetic_kharma_fixture,
)


def nmse(a, b):
    return np.sum((a-b)**2) / np.sum(b*b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-exe', required=True)
    parser.add_argument('--trace-exe')
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve(); work.mkdir(parents=True, exist_ok=True)
    image_exe = str(Path(args.image_exe).resolve())
    paths = [work / f'fluid_{i}.phdf' for i in range(8)]
    for i, path in enumerate(paths):
        write_synthetic_kharma_fixture(path)
        with h5py.File(path, 'r+') as h:
            for key in ('prims.rho', 'prims.u'):
                h[key][...] *= 1 + .03*i
    common = [*synthetic_kharma_transport_args(paths[0]), '--max_steps=20000',
              '--slow_light=1', '--slow_light_observation_time=200',
              '--slow_light_dump_list='+','.join(map(str, paths)),
              '--slow_light_time_list=0,30,60,90,120,150,180,210',
              '--parameter_output=auto']
    for selection in ('coefficients', 'fluid', 'default'):
        method = 'fluid' if selection == 'default' else selection
        modes = common if selection == 'default' else common + [f'--slow_light_interpolation={method}']
        outputs = {}
        for name, extra in (
            ('single', ['--analysis_mode=0']),
            ('fused', ['--freq_list=230000000000,345000000000', '--analysis_mode=0']),
            ('analysis', ['--freq_list=230000000000,345000000000', '--analysis_mode=1']),
        ):
            path = work / f'{selection}_{name}.h5'; outputs[name] = path
            # Trace retains pairwise stepping and has no image step-mode option.
            # Pin only the image driver to that same policy for this comparison.
            run([image_exe, *modes, '--slow_light_step_mode=snapshot',
                 '--slow_light_pipeline=0', '--format=hdf5',
                 '--multifrequency_chunk_size=2', *extra, f'--output={path}'], work)
            with h5py.File(path) as h:
                assert text_attr(h.attrs['slow_light_interpolation']) == method
                assert text_attr(h.attrs['slow_light_step_mode']) == 'snapshot'
                assert np.all(h['frame_0/diagnostics/reason'][...] == 1)
                assert np.max(h['frame_0/freq_0/I'][...]) > 0
                parameters = Path(text_attr(h.attrs['effective_parameter_file']))
                assert f'slow_light_interpolation={method}\n' in parameters.read_text()
                if selection == 'default':
                    with h5py.File(work / f'fluid_{name}.h5') as explicit:
                        for freq in (0,) if name == 'single' else (0, 1):
                            for stokes in 'IQUV':
                                key = f'frame_0/freq_{freq}/{stokes}'
                                np.testing.assert_array_equal(h[key][...], explicit[key][...])
        with h5py.File(outputs['single']) as ref:
            for name in ('fused', 'analysis'):
                with h5py.File(outputs[name]) as h:
                    for stokes in 'IQUV':
                        key = f'frame_0/freq_0/{stokes}'
                        assert nmse(h[key][...], ref[key][...]) < 1e-12, (method, name, stokes)
                    assert np.max(h['frame_0/freq_1/I'][...]) > 0
            if args.trace_exe:
                trace = work / f'{selection}_trace.h5'
                trace_exe = str(Path(args.trace_exe).resolve())
                run([trace_exe, *modes, '--trace_mode=image', '--trace_precision=double',
                     '--trace_compression=0', '--max_trace_samples=4096',
                     '--trace_fields=lambda,coords,plasma,coeffs,stokes', f'--output={trace}'], work)
                with h5py.File(trace) as h:
                    assert text_attr(h.attrs['slow_light_interpolation']) == method
                    assert text_attr(h['parameters/integration'].attrs['slow_light_interpolation']) == method
                    # A one-ulp remainder at a snapshot boundary must not stall
                    # the trace or repeatedly add emission from the same event.
                    assert np.all(h['rays/reason'][...] == 1)
                    assert np.max(h['rays/pass_b_steps'][...]) < 1000
                    for stokes in 'IQUV':
                        a = h[f'rays/final_observed_{stokes}_inv'][...]
                        b = ref[f'frame_0/freq_0/{stokes}_inv'][...].ravel()
                        assert nmse(a, b) < 1e-12, (method, 'trace', stokes)
    for executable in filter(None, (args.image_exe, args.trace_exe)):
        result = run([str(Path(executable).resolve()), '--slow_light_interpolation=primitives'],
                     work, expect_ok=False)
        assert 'must be coefficients or fluid' in result.stderr + result.stdout
        # The default must not enable slow light or break existing fast-light jobs.
        is_trace = executable == args.trace_exe
        reference = None
        for selection in ('default', 'fluid', 'coefficients'):
            output = work / f'fast_{"trace" if is_trace else "image"}_{selection}.h5'
            options = [*synthetic_kharma_transport_args(paths[0]), '--slow_light=0',
                       '--max_steps=20000', f'--output={output}']
            if selection != 'default':
                options.append(f'--slow_light_interpolation={selection}')
            if is_trace:
                options += ['--trace_mode=image', '--trace_precision=double', '--max_trace_samples=4096']
            else:
                options += ['--format=hdf5', '--analysis_mode=0']
            run([str(Path(executable).resolve()), *options], work)
            with h5py.File(output) as h:
                current = [h[f'rays/final_observed_{s}_inv' if is_trace else f'frame_0/freq_0/{s}_inv'][...]
                           for s in 'IQUV']
                assert np.isfinite(current).all() and np.max(current[0]) > 0
                if reference is not None:
                    np.testing.assert_array_equal(current, reference)
                reference = current
    print('PASS: default equals explicit fluid; both methods; frequency/analysis/trace; fast-light unchanged; invalid modes')


if __name__ == '__main__':
    main()
