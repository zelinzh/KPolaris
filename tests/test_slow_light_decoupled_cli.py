#!/usr/bin/env python3
"""Check cache-independent stepping, defaults, aliases and parameter replay."""
import argparse
import itertools
from pathlib import Path

import h5py
import numpy as np

from test_hdf5_contract import (
    run, synthetic_kharma_transport_args, text_attr, write_synthetic_kharma_fixture,
)
from test_slow_light_batch_cli import compare


def check_output(path, mode):
    with h5py.File(path) as output:
        assert text_attr(output.attrs['slow_light_step_mode']) == mode
        assert text_attr(output.attrs['slow_light_interpolation']) == 'fluid'
        assert np.all(output['frame_0/diagnostics/reason'][...] == 1)
        for channel in 'IQUV':
            assert np.isfinite(output[f'frame_0/freq_0/{channel}'][...]).all()
        assert np.max(output['frame_0/freq_0/I'][...]) > 0
        return Path(text_attr(output.attrs['effective_parameter_file']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-exe', required=True)
    parser.add_argument('--analysis-enabled', type=int, default=0)
    parser.add_argument('--max-frequencies', type=int, default=1)
    parser.add_argument('--workdir', required=True, type=Path)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    exe = str(Path(args.image_exe).resolve())
    times = list(range(-60, 271, 30))
    paths = [work / f'fluid_{index}.phdf' for index in range(len(times))]
    for index, path in enumerate(paths):
        write_synthetic_kharma_fixture(path)
        with h5py.File(path, 'r+') as state:
            for name in ('prims.rho', 'prims.u'):
                state[name][...] *= 1 + 0.03 * index
    common = [exe, *synthetic_kharma_transport_args(paths[0]), '--max_steps=20000',
              '--slow_light=1', '--format=hdf5', '--analysis_mode=0',
              '--parameter_output=auto', '--slow_light_prefetch_snapshots=2',
              '--slow_light_dump_list=' + ','.join(map(str, paths)),
              '--slow_light_time_list=' + ','.join(map(str, times))]
    supports = [(200, 0, 10), (201, 1, 11)]
    references = []
    for index, (observer, first, last) in enumerate(supports):
        output = work / f'reference_{index}.h5'
        run([*common, f'--output={output}', '--slow_light_windows_per_block=32',
             '--slow_light_pipeline=0', f'--slow_light_observation_time={observer}',
             '--slow_light_dump_list=' + ','.join(map(str, paths[first:last + 1])),
             '--slow_light_time_list=' + ','.join(map(str, times[first:last + 1]))], work)
        check_output(output, 'decoupled')
        references.append(output)
    for pipeline, block in itertools.product((0, 1), (1, 3, 32)):
        outputs = [work / f'batch_p{pipeline}_b{block}_{index}.h5' for index in range(2)]
        for output in outputs:
            output.unlink(missing_ok=True)
        jobs = work / 'jobs.txt'
        jobs.write_text('\n'.join(f'{observer} {first} {last} "{output}"'
                                 for (observer, first, last), output in zip(supports, outputs)) + '\n')
        run([*common, f'--slow_light_batch_jobs={jobs}', f'--slow_light_windows_per_block={block}',
             f'--slow_light_pipeline={pipeline}'], work)
        for reference, output in zip(references, outputs):
            parameters = check_output(output, 'decoupled')
            compare(reference, output)
            replay = output.with_name(output.stem + '_replay.h5')
            run([exe, f'--parameter_file={parameters}', f'--output={replay}', '--parameter_output=none'], work)
            compare(output, replay)
    single = [*common, '--slow_light_observation_time=200']
    for mode in ('continuous', 'decoupled'):
        output = work / f'alias_{mode}.h5'
        run([*single, f'--slow_light_step_mode={mode}', f'--output={output}'], work)
        parameters = check_output(output, 'decoupled')
        assert 'slow_light_step_mode=decoupled\n' in parameters.read_text()
        compare(references[0], output)
    for pipeline, budget in ((0, 0.00045), (1, 0.00070)):
        output = work / f'budget_p{pipeline}.h5'
        run([*single, f'--slow_light_pipeline={pipeline}', f'--slow_light_snapshot_cache_gib={budget}',
             f'--output={output}'], work)
        compare(references[0], output)
        with h5py.File(output) as result:
            assert result.attrs['slow_light_windows_per_block'] == 3
    fast_reference = None
    for mode in ('default', 'decoupled', 'continuous', 'block', 'snapshot'):
        output = work / f'fast_{mode}.h5'
        command = [exe, *synthetic_kharma_transport_args(paths[0]), '--slow_light=0',
                   '--max_steps=20000', '--format=hdf5', '--analysis_mode=0', f'--output={output}']
        if mode != 'default':
            command.append(f'--slow_light_step_mode={mode}')
        run(command, work)
        with h5py.File(output) as result:
            values = np.stack([result[f'frame_0/freq_0/{channel}'][...] for channel in 'IQUV'])
            assert np.isfinite(values).all() and np.max(values[0]) > 0
            if fast_reference is not None:
                np.testing.assert_array_equal(values, fast_reference)
            fast_reference = values
    workflows = [('multi', ['--freq_list=230000000000,345000000000'])]
    if args.analysis_enabled:
        workflows += [('analysis', ['--analysis_mode=1']),
                      ('multi_analysis', ['--analysis_mode=1', '--freq_list=230000000000,345000000000'])]
    for name, extra in workflows:
        reference = work / f'unified_{name}_full.h5'
        run([*single, *extra, '--slow_light_windows_per_block=32', f'--output={reference}'], work)
        check_output(reference, 'decoupled')
        for block in (1, 3):
            output = work / f'unified_{name}_{block}.h5'
            run([*single, *extra, f'--slow_light_windows_per_block={block}', f'--output={output}'], work)
            check_output(output, 'decoupled')
            compare(reference, output)
    missing = work / 'missing_support.h5'
    missing.unlink(missing_ok=True)
    result = run([*single, f'--output={missing}',
                  '--slow_light_dump_list=' + ','.join(map(str, paths[-2:])),
                  '--slow_light_time_list=' + ','.join(map(str, times[-2:]))], work, expect_ok=False)
    assert 'outside supplied physical time support' in result.stderr
    assert not missing.exists()
    print('PASS: decoupled default; cache/pipeline invariance; aliases; replay; budget; fast light; unified frequency/analysis; coverage')


if __name__ == '__main__':
    main()
