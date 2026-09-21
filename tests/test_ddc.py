#!/usr/bin/env python3
"""DDC STAGE1 input contract and PHDF equivalence; optional real codec integration.

The built-in fixture service tests the wire contract without a codec dependency.
--codec-project additionally encodes an evolving synthetic KHARMA sequence and
runs the actual dense-dump-codec service against its PHDF reconstruction.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

import h5py
import numpy as np
from test_hdf5_contract import write_synthetic_kharma_fixture, synthetic_kharma_transport_args

DATASETS = ('prims.rho', 'prims.u', 'prims.uvec', 'prims.B')
ROOT = Path(__file__).resolve().parents[1]


def wire_frame(path, sequence):
    with h5py.File(path) as h:
        par = h['Input'].attrs['File']
        if isinstance(par, str):
            par = par.encode('utf-8')
        blocks = np.asarray(h['Blocks/loc.lx123'], dtype='<i8')
        arrays = [np.asarray(h[name], dtype='<f4') for name in DATASETS]
        parts = [par, blocks.tobytes(), *(a.tobytes() for a in arrays)]
        fields = ['STAGE1', str(sequence), format(h['Info'].attrs['Time'], '.17g'),
                  str(h['Info'].attrs['NumMeshBlocks']),
                  *(str(x) for x in h['Info'].attrs['MeshBlockSize']),
                  str(len(par)), str(blocks.size), *(str(a.size) for a in arrays),
                  str(sum(len(p) for p in parts))]
    return fields, parts


class FixtureHandler(socketserver.StreamRequestHandler):
    def handle(self):
        command, sequence = self.rfile.readline().decode().strip().split()
        assert command == 'STAGE'
        sequence = int(sequence)
        self.server.requests.append(sequence)
        fields, parts = wire_frame(self.server.frames[sequence], sequence)
        fault = self.server.fault
        if fault == 'timeout':
            time.sleep(1.5)
            return
        if fault == 'sequence': fields[1] = str(sequence + 1)
        elif fault == 'version': fields[0] = 'STAGE2'
        elif fault == 'counts': fields[9] = '1'
        elif fault == 'bytes': fields[13] = '1'
        elif fault == 'time_nan': fields[2] = 'nan'
        elif fault == 'time_mismatch': fields[2] = str(float(fields[2]) + 1)
        elif fault == 'huge_par': fields[7] = str(2**64 - 1)
        elif fault == 'trailing': fields.append('unexpected')
        elif fault == 'nan_primitive':
            a = np.frombuffer(parts[2], dtype='<f4').copy(); a[0] = np.nan
            parts[2] = a.tobytes()
        elif fault == 'duplicate_block':
            a = np.frombuffer(parts[1], dtype='<i8').copy(); a[3:6] = a[:3]
            parts[1] = a.tobytes()
        payload = (' '.join(fields) + '\n').encode() + b''.join(parts)
        if fault == 'truncated': payload = payload[:-9]
        if fault == 'error': payload = b'ERROR fixture decode failure\n'
        try:
            # Deliberately split the stream inside header / arrays.
            for i in range(0, len(payload), 97):
                self.wfile.write(payload[i:i+97])
        except (BrokenPipeError, ConnectionResetError):
            pass  # Expected when the client rejects an invalid header.


@contextlib.contextmanager
def fixture_service(frames, socket_path):
    with socketserver.ThreadingUnixStreamServer(str(socket_path), FixtureHandler) as server:
        server.frames, server.requests, server.fault = frames, [], None
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05})
        thread.start()
        try:
            yield server
        finally:
            server.shutdown(); thread.join()
            socket_path.unlink(missing_ok=True)


def compare_products(a, b, trace=False):
    count = 0
    with h5py.File(a) as ha, h5py.File(b) as hb:
        def visit(name, obj):
            nonlocal count
            if not isinstance(obj, h5py.Dataset): return
            if not (name.startswith('frame_') or (trace and name.startswith('trace/'))): return
            av, bv = obj[...], hb[name][...]
            assert av.shape == bv.shape, name
            assert np.array_equal(av, bv, equal_nan=True), f'{name} differs: {a.name}, {b.name}'
            count += 1
        ha.visititems(visit)
        assert count > 0
        if not trace:
            assert np.all(ha['frame_0/diagnostics/reason'][...] == 1)
            assert float(np.sum(ha['frame_0/freq_0/I_inv'][...])) > 0
    return count


def run(exe, base, work, label, extra=(), env=None, fail=None, trace=False):
    output = work / (label + '.h5')
    args = [str(exe), *base, *extra, '--parameter_output=auto', f'--output={output}']
    if not trace: args.append('--format=hdf5')
    start = time.monotonic()
    result = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
    elapsed = time.monotonic() - start
    (work / (label + '.log')).write_text(result.stdout + result.stderr)
    (work / (label + '.command.json')).write_text(json.dumps(args, indent=2) + '\n')
    if fail:
        assert result.returncode != 0 and fail.lower() in (result.stdout + result.stderr).lower(), (label, result.stdout[-1000:], result.stderr[-2000:])
    else:
        assert result.returncode == 0, (label, result.stdout[-1000:], result.stderr[-2000:])
    return output, elapsed


def compare_suite(image, trace, frames, sock, work, prefix, manifest=None):
    virtual = [work / f'ddc_frame_{i:05d}.phdf' for i in range(len(frames))]
    base = synthetic_kharma_transport_args(frames[1])
    native = [f'--kharma_dump={virtual[1]}', '--kharma_ddc_native=1', f'--kharma_ddc_socket={sock}']
    if manifest: native.append(f'--kharma_ddc_manifest={manifest}')
    freq = ['--freq_list=230000000000,345000000000']
    times = [str(float(h5_time(p))) for p in frames]
    slow = ['--slow_light=1', '--slow_light_observation_time=200', '--slow_light_time_list='+','.join(times)]
    raw_list = '--slow_light_dump_list='+','.join(map(str, frames))
    ddc_list = '--slow_light_dump_list='+','.join(map(str, virtual))
    selection = ['--equatorial_h_over_r=.2', '--direct_only=1', '--faraday_rotation=0']
    analysis = ['--analysis_mode=1', '--analysis_partition=region', '--analysis_response=density_scale']
    cases = [('fast', []), ('fast_multi', freq), ('fast_analysis_selection', selection+analysis),
             ('slow_multi', slow+freq+['--slow_light_prefetch=0']), ('slow_prefetch', slow+freq+['--slow_light_prefetch=1']),
             ('slow_analysis_selection', slow+selection+analysis),
             ('slow_implicit_time', ['--slow_light=1', '--slow_light_observation_time=200'])]
    rows = []
    for label, settings in cases:
        is_slow = label.startswith('slow')
        raw, raw_seconds = run(image, base, work, prefix+'_'+label+'_phdf', [*settings, '--kharma_ddc_native=0', *([raw_list] if is_slow else [])])
        decoded, native_seconds = run(image, base, work, prefix+'_'+label+'_native', [*settings, *native, *([ddc_list] if is_slow else [])])
        count = compare_products(raw, decoded)
        with h5py.File(decoded) as h:
            for obj in (h, h['parameters/radiation']):
                assert obj.attrs['kharma_ddc_native'] == 1
                assert 'STAGE1' in str(obj.attrs['kharma_ddc_protocol'])
        rows.append(dict(case=label, exact_datasets=count, stokes_nmse=0., phdf_seconds=raw_seconds, native_seconds=native_seconds))
        if label == 'fast':
            repeat, _ = run(image, [], work, prefix+'_replay', ['--parameter_file='+str(decoded)+'.params'])
            compare_products(decoded, repeat)
            # Legacy wrappers supply environment variables, CLI can disable them.
            env = dict(os.environ, KPOLARIS_DDC_NATIVE='1', KPOLARIS_DDC_SOCKET=str(sock))
            legacy, _ = run(image, base, work, prefix+'_environment', [f'--kharma_dump={virtual[1]}'], env=env)
            compare_products(decoded, legacy)
            override, _ = run(image, base, work, prefix+'_disable', ['--kharma_ddc_native=0'], env=env)
            compare_products(raw, override)
    if trace:
        trace_base = [x for x in base if x != '--timing=0'] + ['--trace_mode=single', '--ix=0', '--iy=0', '--trace_fields=all', '--trace_precision=double']
        for label, settings in [('fast', []), ('slow', slow)]:
            a, _ = run(trace, trace_base, work, prefix+'_trace_'+label+'_phdf', [*settings, '--kharma_ddc_native=0', *([raw_list] if settings else [])], trace=True)
            b, _ = run(trace, trace_base, work, prefix+'_trace_'+label+'_native', [*settings, *native, *([ddc_list] if settings else [])], trace=True)
            rows.append(dict(case='trace_'+label, exact_datasets=compare_products(a, b, trace=True)))
    return rows, base, native, slow+[ddc_list]


def h5_time(path):
    with h5py.File(path) as h: return h['Info'].attrs['Time']


@contextlib.contextmanager
def codec_service(project, manifest, socket_path, work):
    ready = work/'service.ready.json'; ready.unlink(missing_ok=True)
    stats = work/'service.stats.json'
    args = [sys.executable, str(project/'scripts/serve_ddc_for_kpolaris.py'), '--manifest', str(manifest),
            '--socket', str(socket_path), '--cache-dir', str(work/'cache'), '--maximum-cache-files', '4',
            '--prefetch-files', '2', '--native-archive-cache-chunks', '8',
            '--ready-file', str(ready), '--stats-file', str(stats)]
    with (work/'service.log').open('w') as log:
        process = subprocess.Popen(args, cwd=project, stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+60
            while not ready.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('DDC service startup failed: '+str(work/'service.log'))
                time.sleep(.05)
            yield
        finally:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill(); process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--codec-project', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='kp-ddc-') as temporary:
        work = (args.workdir or Path(temporary)/'work').resolve(); work.mkdir(parents=True, exist_ok=True)
        sock = Path(temporary)/'s.sock'
        frames = []
        raw_dir = work/'raw'; raw_dir.mkdir(exist_ok=True)
        for i, t in enumerate([0., 100., 1000.]):
            path = raw_dir/f'tiny.out0.{i:05d}.phdf'
            write_synthetic_kharma_fixture(path, meshblocks_x1=2)
            with h5py.File(path, 'r+') as h:
                h['Info'].attrs['Time'] = t
                for j, name in enumerate(DATASETS):
                    values = h[name][...]
                    values *= 1 + (.08 + .02*j)*i*i
                    h[name][...] = values
            frames.append(path)
        image = args.image.resolve(); trace = args.trace.resolve() if args.trace else None
        with fixture_service(frames, sock) as server:
            rows, base, native, slow = compare_suite(image, trace, frames, sock, work, 'wire')
            faults = dict(sequence='sequence', version='header', counts='counts', bytes='byte count', time_nan='header', huge_par='parameter text', trailing='trailing', truncated='payload', error='fixture decode failure', nan_primitive='nonfinite', duplicate_block='duplicate', time_mismatch='time disagrees', timeout='receive DDC native')
            for fault, error in faults.items():
                server.fault = fault
                run(image, base, work, 'reject_'+fault, native+(slow if fault=='time_mismatch' else [])+(['--kharma_ddc_timeout_seconds=1'] if fault=='timeout' else []), fail=error)
            server.fault = None
            repeated = [work/'ddc_frame_00000.phdf', work/'ddc_frame_00000.phdf', work/'ddc_frame_00002.phdf']
            run(image, base, work, 'reject_repeated_sequence', [*native, *slow, '--slow_light_dump_list='+','.join(map(str, repeated))], fail='more than once')
            for name in ('ddc_frame_-1.phdf', 'ddc_frame_1x.phdf', 'ddc_frame_2147483648.phdf'):
                run(image, base, work, 'reject_'+name, [*native, '--kharma_dump='+str(work/name)], fail='DDC frame')
            # Keep the absent endpoint short even when CTest's build path is long.
            run(image, base, work, 'reject_unavailable', [*native, '--kharma_ddc_socket='+str(Path(temporary)/'absent.sock')], fail='connect')
            run(image, base, work, 'reject_long_socket', [*native, '--kharma_ddc_socket='+str(Path(temporary)/('x'*128+'.sock'))], fail='path is too long')
            for setting in ('native=2', 'timeout_seconds=0', 'timeout_seconds=1x'):
                run(image, base, work, 'reject_option_'+setting.replace('=','_'), [*native, '--kharma_ddc_'+setting], fail='kharma_ddc')
            # Explicit native mode must not read an unrelated stale cache file.
            stale = work/'ddc_frame_00001.phdf'; stale.write_bytes(b'not an HDF5 file')
            try:
                a, _ = run(image, base, work, 'stale_native', native)
                compare_products(work/'wire_fast_phdf.h5', a)
            finally: stale.unlink()
            request_count = len(server.requests)
        report = dict(all_passed=True, fixture_wire=rows, rejection_cases=len(faults)+9, stage_requests=request_count)
        if args.codec_project:
            project = args.codec_project.resolve()
            codec = work/'codec'
            command = [sys.executable, str(project/'scripts/encode_dense_sequence.py'), '--segment-dir', str(raw_dir), '--output-dir', str(codec), '--keyframe-stride', '2', '--datasets', ','.join(DATASETS), '--bits', '8', '--archive-backend', 'channel-bzip2-adaptive', '--keyframe-backend', 'raw']
            result = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=180)
            (work/'encode.log').write_text(result.stdout+result.stderr)
            assert result.returncode == 0, result.stderr
            manifest = codec/'sequence_manifest.json'
            sys.path.insert(0, str(project)); sys.path.insert(0, str(project/'src'))
            from scripts.decode_dense_sequence import decode_sequence_frame
            decoded = []
            for i in range(3):
                path = work/f'decoded_{i}.phdf'; decode_sequence_frame(manifest, i, path, overwrite=True); decoded.append(path)
            with codec_service(project, manifest, sock, work):
                rows, _, _, _ = compare_suite(image, trace, decoded, sock, work, 'codec', manifest)
            # Codec loss is a separate comparison, never inferred from transport equality.
            original, _ = run(image, synthetic_kharma_transport_args(frames[1]), work, 'codec_loss_original', ['--kharma_ddc_native=0'])
            reconstructed, _ = run(image, synthetic_kharma_transport_args(decoded[1]), work, 'codec_loss_reconstructed', ['--kharma_ddc_native=0'])
            nmse = {}
            with h5py.File(original) as a, h5py.File(reconstructed) as b:
                for stokes in 'IQUV':
                    x = a[f'frame_0/freq_0/{stokes}_inv'][...]; y = b[f'frame_0/freq_0/{stokes}_inv'][...]
                    denominator = float(np.sum(x*x))
                    nmse[stokes] = float(np.sum((x-y)**2)/denominator) if denominator else None
            report['codec'] = dict(cases=rows, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), compression_image_nmse=nmse)
            assert not list((work/'cache').glob('*.phdf')), 'Native service materialized PHDF files'
        (work/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(f'DDC tests passed; {work}/summary.json')


if __name__ == '__main__':
    main()
