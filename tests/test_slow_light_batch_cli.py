#!/usr/bin/env python3
"""Check the mainline KHARMA batch CLI against individual slow-light images."""
import argparse
import itertools
from pathlib import Path
import h5py
import numpy as np
from test_hdf5_contract import (
    write_synthetic_kharma_fixture, synthetic_kharma_transport_args,
    run, check_image_contract, check_analysis_contract,
)


def compare(reference, candidate):
    with h5py.File(reference) as a, h5py.File(candidate) as b:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset) or not name.startswith('frame_0/'):
                return
            av, bv = obj[...], b[name][...]
            assert np.all(np.isfinite(bv)), name
            assert np.array_equal(av, bv), f'{candidate}: {name}'
        a.visititems(visit)
        for key in ('adaptive_tolerance', 'max_step', 'slow_light_observation_time',
                    'analysis_mode', 'direct_only', 'equatorial_h_over_r', 'frequency_hz', 'slow_light_interpolation', 'slow_light_step_mode'):
            assert a.attrs[key] == b.attrs[key], key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-exe', required=True)
    p.add_argument('--interpolation', choices=('default','coefficients','fluid'), default='default')
    p.add_argument('--workdir', type=Path, required=True)
    args = p.parse_args()
    exe = str(Path(args.image_exe).resolve())
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    fixture = work / 'fixture.phdf'
    write_synthetic_kharma_fixture(fixture)
    # Different snapshots make accidental reuse of a single fluid state visible.
    paths = []
    for i in range(9):
        dest = work / f'fluid_{i}.phdf'
        write_synthetic_kharma_fixture(dest)
        with h5py.File(dest, 'r+') as f:
            for name in ('prims.rho', 'prims.u'):
                f[name][...] *= 1 + 0.03 * i
        paths.append(dest)
    times = list(range(0, 241, 30))
    supports = [(200, 0, 7), (201, 1, 8)]
    common = [exe, *synthetic_kharma_transport_args(fixture), '--slow_light=1',
              '--format=hdf5', '--slow_light_pipeline=0', '--timing=0', '--parameter_output=auto', '--max_steps=20000',
              '--slow_light_step_mode=snapshot']
    if args.interpolation != 'default':
        common.append(f'--slow_light_interpolation={args.interpolation}')
    expected_method = 'fluid' if args.interpolation == 'default' else args.interpolation
    for analysis in (0, 1):
        references = []
        for j, (observer, first, last) in enumerate(supports):
            dest = work / f'reference_a{analysis}_{j}.h5'
            run([*common, f'--analysis_mode={analysis}', f'--output={dest}',
                 f'--slow_light_observation_time={observer}',
                 '--slow_light_dump_list=' + ','.join(str(x) for x in paths[first:last+1]),
                 '--slow_light_time_list=' + ','.join(str(x) for x in times[first:last+1])], work)
            with h5py.File(dest) as h:
                stored_method = h.attrs['slow_light_interpolation']
                if isinstance(stored_method, bytes):
                    stored_method = stored_method.decode()
                assert stored_method == expected_method
                assert np.all(h['frame_0/diagnostics/reason'][...] == 1), 'reference failed to return to camera'
                assert np.max(h['frame_0/diagnostics/steps'][...]) < 20000
                assert np.max(h['frame_0/freq_0/I'][...]) > 0
            references.append(dest)
        for pipeline, block in itertools.product((0,1),(1,3,32)):
            dests = [work / f'batch_a{analysis}_p{pipeline}_b{block}_{j}.h5' for j in range(2)]
            for dest in dests:
                dest.unlink(missing_ok=True)
            jobs = work / 'jobs.txt'
            jobs.write_text('\n'.join(f'{obs} {first} {last} "{dest}"'
                                      for (obs, first, last), dest in zip(supports, dests)) + '\n')
            result = run([*common, f'--analysis_mode={analysis}', f'--slow_light_batch_jobs={jobs}',
                          f'--slow_light_windows_per_block={block}', f'--slow_light_pipeline={pipeline}', '--slow_light_prefetch_snapshots=2',
                          '--slow_light_dump_list=' + ','.join(map(str, paths)),
                          '--slow_light_time_list=' + ','.join(map(str, times))], work)
            for j, dest in enumerate(dests):
                check_image_contract(dest)
                if analysis:
                    check_analysis_contract(dest)
                compare(references[j], dest)
                with h5py.File(dest) as f:
                    assert f.attrs['slow_light_batch_size'] == 2
                    assert f.attrs['slow_light_batch_index'] == j
                    assert f.attrs['slow_light_windows_per_block'] == block
                    params = Path(f.attrs['effective_parameter_file'].decode()
                                  if isinstance(f.attrs['effective_parameter_file'], bytes)
                                  else f.attrs['effective_parameter_file'])
                assert params.is_file()
                # A per-frame parameter file must rerun one frame, not its batch.
                rerun = work / f'rerun_a{analysis}_p{pipeline}_b{block}_{j}.h5'
                run([exe, f'--parameter_file={params}', f'--output={rerun}', '--parameter_output=none'], work)
                compare(dest, rerun)
    # Byte budget chooses three windows on the 8^3 double-precision fixture;
    # an explicit smaller window limit takes precedence. The budget includes
    # the retained first model, but not CUDA runtime workspace.
    budget_common = [*common, '--analysis_mode=0', '--slow_light_observation_time=200',
                     '--slow_light_dump_list=' + ','.join(map(str, paths[:8])),
                     '--slow_light_time_list=' + ','.join(map(str, times[:8])),
                     '--slow_light_snapshot_cache_gib=0.00025', '--slow_light_prefetch_snapshots=2']
    for pipeline, explicit in itertools.product((0,1),(False,True)):
        output = work / f'cache_p{pipeline}_{int(explicit)}.h5'
        cmd = [*budget_common, f'--output={output}', f'--slow_light_pipeline={pipeline}']
        if pipeline: cmd.append('--slow_light_snapshot_cache_gib=0.00050')
        if explicit:
            cmd.append('--slow_light_windows_per_block=2')
        run(cmd, work)
        compare(work / 'reference_a0_0.h5', output)
        with h5py.File(output) as f:
            assert f.attrs['slow_light_windows_per_block'] == (2 if explicit else 3)
    # Failure while preparing the next block must release the active kernel's
    # owners safely and must not leave a successful image behind.
    missing_paths=list(paths[:8]); missing_paths[3]=work/'missing_next_block.phdf'
    missing_output=work/'missing_input.h5'; missing_output.unlink(missing_ok=True)
    failure=run([*common, '--analysis_mode=0', '--slow_light_pipeline=1',
                 '--slow_light_windows_per_block=1', '--slow_light_prefetch_snapshots=2',
                 '--slow_light_observation_time=200', f'--output={missing_output}',
                 '--slow_light_dump_list='+','.join(map(str,missing_paths)),
                 '--slow_light_time_list='+','.join(map(str,times[:8]))],work,expect_ok=False)
    assert not missing_output.exists()
    assert 'illegal memory access' not in failure.stderr.lower()
    assert 'missing_next_block' in failure.stderr or 'HDF5' in failure.stderr
    print('PASS: snapshot byte budget and explicit window cap, one/two banks')
    print('PASS: KHARMA varying snapshots; CLI blocks 1/3/32; IQUV and every diagnostic dataset; independent per-frame reruns')

    block_common = [arg for arg in common if not arg.startswith('--slow_light_step_mode=')]
    block_common.append('--slow_light_step_mode=block')
    for analysis, block in itertools.product((0, 1), (1, 3, 32)):
        dests = [work / f'block_a{analysis}_b{block}_{j}.h5' for j in range(2)]
        for dest in dests:
            dest.unlink(missing_ok=True)
        jobs = work / 'block_jobs.txt'
        jobs.write_text('\n'.join(f'{obs} {first} {last} "{dest}"'
                                  for (obs, first, last), dest in zip(supports, dests)) + '\n')
        run([*block_common, f'--analysis_mode={analysis}',
             f'--slow_light_batch_jobs={jobs}', f'--slow_light_windows_per_block={block}',
             '--slow_light_pipeline=1', '--slow_light_prefetch_snapshots=2',
             '--slow_light_dump_list=' + ','.join(map(str, paths)),
             '--slow_light_time_list=' + ','.join(map(str, times))], work)
        for j, dest in enumerate(dests):
            check_image_contract(dest)
            if analysis: check_analysis_contract(dest)
            with h5py.File(dest) as h:
                mode = h.attrs['slow_light_step_mode']
                if isinstance(mode, bytes): mode = mode.decode()
                assert mode == 'block'
                assert np.all(h['frame_0/diagnostics/reason'][...] == 1)
                params = h.attrs['effective_parameter_file']
                if isinstance(params, bytes): params = params.decode()
            replay = work / f'block_replay_a{analysis}_b{block}_{j}.h5'
            run([exe, f'--parameter_file={params}', f'--output={replay}', '--parameter_output=none'], work)
            compare(dest, replay)
    failure = run([*common, '--slow_light_step_mode=invalid'], work, expect_ok=False)
    assert 'slow_light_step_mode must be decoupled, block or snapshot' in failure.stderr
    print('PASS: legacy block stepping, offset batch supports, diagnostics, exact explicit-block saved-parameter replays, invalid mode')


if __name__ == '__main__':
    main()
