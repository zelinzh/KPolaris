#!/usr/bin/env python3
"""Exercise each fluid backend through the unified slow-light image scheduler."""
import argparse
from pathlib import Path

import h5py
import numpy as np

import test_hdf5_contract as fixtures
from test_slow_light_batch_cli import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-exe', required=True, type=Path)
    parser.add_argument('--model', required=True, choices=('iharm', 'kharma', 'athenak', 'bhac', 'hamr'))
    parser.add_argument('--analysis-enabled', type=int, default=1)
    parser.add_argument('--workdir', required=True, type=Path)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    writer = getattr(fixtures, f'write_synthetic_{args.model}_fixture')
    make_args = getattr(fixtures, f'synthetic_{args.model}_transport_args')
    paths = [work / f'state_{i}' for i in range(10)]
    for i, path in enumerate(paths):
        writer(path)
        # Vary actual primitive fields, rather than just declaring new times.
        if args.model in ('iharm', 'kharma'):
            with h5py.File(path, 'r+') as state:
                if args.model == 'kharma':
                    for key in ('prims.rho', 'prims.u'):
                        state[key][...] *= 1 + .03 * i
                elif args.model == 'iharm':
                    values = state['prims'][...]
                    values[..., :2] *= 1 + .03 * i
                    state['prims'][...] = values
        elif args.model == 'athenak':
            raw = bytearray(path.read_bytes())
            values = np.frombuffer(raw, dtype='<f4', offset=len(raw) - 8*8**3*4).reshape(8, -1)
            values[[0, 4]] *= 1 + .03*i
            path.write_bytes(raw)
        elif args.model == 'bhac':
            raw = bytearray(path.read_bytes())
            values = np.frombuffer(raw, dtype='<f8', count=13*8**3).reshape(13, -1)
            values[[0, 12]] *= 1 + .03*i
            path.write_bytes(raw)
        else:
            data = path / 'new_dump0000'
            values = np.frombuffer(data.read_bytes(), dtype='<f4').copy().reshape(-1, 9)
            values[:, :2] *= 1 + .03*i
            data.write_bytes(values.tobytes())
    times = [-100, -63, -20, 0, 31, 60, 105, 150, 200, 260]
    common = [str(args.image_exe.resolve()), *make_args(paths[0]), '--slow_light=1',
              '--slow_light_observation_time=200', '--max_steps=20000', '--format=hdf5',
              '--parameter_output=auto', '--slow_light_dump_list=' + ','.join(map(str, paths)),
              '--slow_light_time_list=' + ','.join(map(str, times)),
              '--freq_list=230000000000,345000000000,86000000000',
              '--slow_light_prefetch_snapshots=2', '--slow_light_pipeline=0']
    for method in ('fluid', 'coefficients'):
        for analysis in range(args.analysis_enabled + 1):
            extra = [f'--slow_light_interpolation={method}', f'--analysis_mode={analysis}']
            if analysis:
                extra += ['--analysis_radial_bins=3', '--analysis_partition=radial',
                          '--analysis_response=density_scale']
            reference = work / f'{method}_a{analysis}_full.h5'
            fixtures.run([*common, *extra, '--slow_light_windows_per_block=32', f'--output={reference}'], work)
            with h5py.File(reference) as result:
                assert fixtures.text_attr(result.attrs['slow_light_step_mode']) == 'decoupled'
                assert np.all(result['frame_0/diagnostics/reason'][...] == 1)
                for f in range(3):
                    assert np.max(result[f'frame_0/freq_{f}/I'][...]) > 0
            for block in (1, 3):
                candidate = work / f'{method}_a{analysis}_b{block}.h5'
                fixtures.run([*common, *extra, f'--slow_light_windows_per_block={block}',
                              '--slow_light_step_mode=decoupled', f'--output={candidate}'], work)
                compare(reference, candidate)
            # Fixed geometric cap and no frequency-dependent radiation cap:
            # fused and separate solves must have identical IQUV and diagnostics.
            geometric = ['--max_radiation_depth=0', '--max_absorption_depth=0', '--max_faraday_depth=0',
                         '--max_radiation_step=0.1', '--max_step=0.1', '--slow_light_windows_per_block=3']
            fused, separate = (work / f'{method}_a{analysis}_{name}.h5' for name in ('fused', 'separate'))
            fixtures.run([*common, *extra, *geometric, '--multifrequency_chunk_size=0', f'--output={fused}'], work)
            fixtures.run([*common, *extra, *geometric, '--multifrequency_chunk_size=1', f'--output={separate}'], work)
            compare(fused, separate)
            with h5py.File(fused) as combined:
                for f, freq in enumerate((230e9, 345e9, 86e9)):
                    single = work / f'{method}_a{analysis}_single{f}.h5'
                    command = [x for x in common if not x.startswith('--freq_list=')]
                    fixtures.run([*command, *extra, *geometric, f'--freq={freq}', f'--output={single}'], work)
                    with h5py.File(single) as result:
                        def visit(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                np.testing.assert_array_equal(obj[...], combined[f'frame_0/freq_{f}/{name}'][...],
                                                              err_msg=f'{args.model}/{method}/{f}/{name}')
                        result['frame_0/freq_0'].visititems(visit)
    # The same scheduler must also compose observer-time batches with spectra
    # and analysis, and each output's effective parameters must replay alone.
    outputs = [work / f'batch_{j}.h5' for j in range(2)]
    for output in outputs:
        output.unlink(missing_ok=True)
    jobs = work / 'jobs.txt'
    jobs.write_text(''.join(f'{200+j} {j} 9 "{output}"\n' for j, output in enumerate(outputs)))
    batch_options = [f'--analysis_mode={args.analysis_enabled}', '--analysis_radial_bins=3',
                     '--slow_light_windows_per_block=3']
    fixtures.run([*common, *batch_options, f'--slow_light_batch_jobs={jobs}'], work)
    for output in outputs:
        with h5py.File(output) as result:
            parameters = Path(fixtures.text_attr(result.attrs['effective_parameter_file']))
        replay = output.with_name(output.stem + '_replay.h5')
        fixtures.run([str(args.image_exe.resolve()), f'--parameter_file={parameters}',
                      '--parameter_output=none', f'--output={replay}'], work)
        compare(output, replay)
    # Probe is geometric. Accept either label with identical numerical output.
    probe = [x for x in common if not x.startswith('--freq_list=')]
    results = []
    for mode in ('decoupled', 'snapshot', 'block'):
        result = fixtures.run([*probe, '--slow_light_time_probe=1', f'--slow_light_step_mode={mode}'], work)
        lines = result.stdout.splitlines()
        start = lines.index('KPolaris slow-light time probe')
        results.append(lines[start:])
        assert any(line.startswith('required_time_min ') for line in results[-1])
        assert any(line.startswith('mean_pass_b_steps ') for line in results[-1])
        assert 'uses snapshot stepping' not in result.stderr
    assert results[0] == results[1] == results[2]
    print(f'PASS: {args.model}; both interpolations; cache invariance; three frequencies; diagnostics; independent solves; geometric probe')


if __name__ == '__main__':
    main()
