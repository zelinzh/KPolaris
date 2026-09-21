#!/usr/bin/env python3
"""Bounded intra-archive scheduling for the external dense-dump-codec decoder.

The codec remains the authority for reconstruction, archive validation and
keyframe resolution. This adapter shares immutable anchors and schedules its
decode contract by dataset and compressed time chunk. A verified NumPy path
reduces temporary arrays; unknown codec revisions use decode_dataset directly.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from collections import defaultdict, OrderedDict, Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from kpolaris_ddc_reconstruct import Reconstruction


def load_codec(project):
    project = Path(project).resolve()
    for directory in (project / 'src', project / 'scripts'):
        if not directory.is_dir():
            raise ValueError(f'Missing codec directory: {directory}')
    sys.path[:0] = [str(project / 'scripts'), str(project / 'src')]
    modules = tuple(importlib.import_module(name) for name in ('decode_dense_sequence', 'decode_dump_codec'))
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(project):
            raise RuntimeError(f'Codec module already loaded from a different project: {module.__file__}')
    return modules


class ParallelSequenceDecoder:
    """One service caller; at most workers dataset/chunk tasks at once.

    Each task owns its archive handle and LRU. No archive object, HDF5 handle,
    mutable decoded array or codec cache is shared between worker threads.
    """
    datasets = ('prims.rho', 'prims.u', 'prims.uvec', 'prims.B')

    def __init__(self, codec_project, manifest, *, workers=8, cache_chunks=8, scheme=None, reconstruction='auto'):
        if workers < 1 or cache_chunks < 0:
            raise ValueError('workers must be positive and cache_chunks nonnegative')
        self.sequence, self.codec = load_codec(codec_project)
        self.manifest_path = Path(manifest)
        self.manifest = self.sequence.load_manifest(self.manifest_path)
        self.workers, self.cache_chunks, self.scheme = workers, cache_chunks, scheme
        self.reconstruction = Reconstruction(self.codec, reconstruction)
        self._anchors = OrderedDict()
        self.times = {int(row['sequence']): float(row['time']) for row in self.manifest['frames']}
        self.keyframes = {self.sequence.phdf_sequence(Path(p)): Path(p) for p in self.manifest['keyframes']}

    @staticmethod
    def _anchor_key(path):
        path = Path(path)
        if not path.is_file():
            return None
        stat = path.stat()
        return str(path.resolve()), stat.st_size, stat.st_mtime_ns

    def _remember_anchor(self, key, native, raw):
        transformed = {d: self.codec.transformed(a, d) for d, a in raw.items()}
        for a in (*raw.values(), *transformed.values()):
            a.flags.writeable = False
        value = native, raw, transformed
        if key is not None:
            self._anchors[key] = value
            self._anchors.move_to_end(key)
            while len(self._anchors) > 4:
                self._anchors.popitem(last=False)
        return value

    def _keyframe_results(self, keys):
        # Retained raw keyframes can be both output frames and immutable GOP
        # anchors. Reuse their arrays; dependency-chain resolution stays external.
        results, missing = {}, []
        for seq in keys:
            path = self.sequence.existing_raw_keyframe(self.manifest, seq, self.keyframes[seq])
            key = self._anchor_key(path) if path is not None else None
            if key is not None and key in self._anchors:
                self._anchors.move_to_end(key)
                native, raw, _ = self._anchors[key]
                results[seq] = {**native, 'sequence': seq, 'time': self.times[seq],
                                'datasets': dict(raw), 'source_phdf': str(path), 'exact_keyframe': True}
            else:
                missing.append(seq)
        if missing:
            fresh = self.sequence.decode_sequence_frames_to_arrays(
                self.manifest_path, tuple(missing), requested_scheme=self.scheme,
                workers=1, archive_cache_chunks=self.cache_chunks)
            for seq, frame in fresh.items():
                path = frame.get('source_phdf')
                key = self._anchor_key(path) if path is not None else None
                # Temporary materialized keyframes have already been removed by
                # the reference decoder; never retain a cache entry for them.
                if key is not None:
                    native = {k: frame[k] for k in ('par_text','num_meshblocks','meshblock_size','block_order')}
                    self._remember_anchor(key, native, frame['datasets'])
            results.update(fresh)
        return results

    def __call__(self, sequences, *, on_frame=None):
        import numpy as np
        requested = tuple(dict.fromkeys(map(int, sequences)))
        if not requested:
            return {}
        for seq in requested:
            if seq not in self.times:
                raise KeyError(f'Sequence {seq} is absent from the manifest')
        # Preserve the codec's exact keyframe path, including dependency chains.
        keys = tuple(seq for seq in requested if seq in self.keyframes)
        results = self._keyframe_results(keys)
        middle = [seq for seq in requested if seq not in self.keyframes]
        if on_frame:
            for seq in keys:
                on_frame(seq, results.pop(seq))
        if not middle:
            return results
        scheme_name, scheme = self.sequence.select_scheme(self.manifest, self.scheme)
        groups = self.sequence.group_archive_frames(scheme['archive_paths'], middle)
        with tempfile.TemporaryDirectory(prefix='kpolaris_ddc_anchors_') as tmp, \
                ThreadPoolExecutor(max_workers=self.workers) as pool:
            tmp = Path(tmp)
            keyframe_cache, active = {}, set()
            tasks = []
            for path, rows in groups.items():
                metadata = self.sequence.read_archive_metadata_with_retry(path)
                anchors = []
                for field in ('start_file', 'end_file'):
                    preferred = Path(metadata[field])
                    seq = self.sequence.phdf_sequence(preferred)
                    anchor = self.sequence.existing_raw_keyframe(self.manifest, seq, preferred)
                    if anchor is None:
                        anchor = tmp / f'keyframe_{seq:05d}.phdf'
                        self.sequence._materialize_keyframe(
                            self.manifest, seq, anchor, overwrite=True,
                            temporary_path=tmp, cache=keyframe_cache, active=active)
                    anchors.append(anchor)
                h5py = self.codec.require_h5py()
                # Read and close HDF5 serially; worker threads use only arrays.
                # Bounded immutable anchor cache, shared by adjacent GOPs.
                # Materialized temporary anchors are call-local and not retained.
                def read_anchor(path):
                    key = self._anchor_key(path)
                    cache = getattr(self, '_anchors', {})
                    if key in cache:
                        cache.move_to_end(key)
                        return cache[key]
                    with h5py.File(path, 'r') as handle:
                        native = self.codec.read_kharma_native_metadata(handle)
                        raw = {d: np.asarray(handle[d][...], dtype=np.float32) for d in self.datasets}
                    return self._remember_anchor(key if not path.is_relative_to(tmp) else None, native, raw)
                native, start, start_t = read_anchor(anchors[0])
                _, end, end_t = read_anchor(anchors[1])
                arrays = {d: (start[d], end[d]) for d in self.datasets}
                transformed = {d: (start_t[d], end_t[d]) for d in self.datasets}
                with self.sequence.open_archive(path, cache_chunks=0) as archive:
                    # Match the codec's (frame_index - 1) // chunk_frames groups.
                    # ZIP archives have no channel index and use one frame/job.
                    chunk_frames = max(1, int(getattr(archive, 'container', {}).get('chunk_frames', 1)))
                    taus = {i: self.codec.frame_tau(archive, metadata, self.datasets, i) for i in rows}
                chunks = defaultdict(list)
                for i in rows:
                    chunks[(i - 1) // chunk_frames].append(i)
                # Keep read-only anchors with each task, so the same worker pool
                # stays populated across archive boundaries instead of draining
                # to the slowest dataset at the end of every small GOP.
                context = (path, metadata, arrays, transformed, rows)
                tasks.extend((context, d, tuple(indices))
                             for indices in chunks.values() for d in self.datasets)
                for i, seq in rows.items():
                    results[seq] = {**native, 'sequence': seq, 'time': self.times[seq],
                                    'tau': float(taus[i]), 'datasets': {},
                                    'archive': str(path), 'frame_index': i,
                                    'exact_keyframe': False, 'scheme': scheme_name}

            def decode_task(task):
                (path, metadata, arrays, transformed, rows), dataset, indices = task
                with self.sequence.open_archive(path, cache_chunks=self.cache_chunks) as archive:
                    decoded = [(rows[i], np.ascontiguousarray((self.reconstruction.decode if hasattr(self, 'reconstruction') else self.codec.decode_dataset)(
                        archive, metadata, dataset, i,
                        start=arrays[dataset][0], end=arrays[dataset][1],
                        start_transformed=transformed[dataset][0],
                        end_transformed=transformed[dataset][1]), dtype='<f4')) for i in indices]
                return dataset, decoded

            # Bounded submissions also bound completed-but-unconsumed arrays.
            pending = set()
            iterator = iter(tasks)
            for _ in range(min(self.workers, len(tasks))):
                pending.add(pool.submit(decode_task, next(iterator)))
            try:
                while pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        dataset, decoded = future.result()
                        for seq, value in decoded:
                            results[seq]['datasets'][dataset] = value
                            if on_frame and len(results[seq]['datasets']) == len(self.datasets):
                                on_frame(seq, results.pop(seq))
                        task = next(iterator, None)
                        if task is not None:
                            pending.add(pool.submit(decode_task, task))
            except BaseException:
                for future in pending:
                    future.cancel()
                # The executor waits before anchors/temporary files disappear.
                raise
        return results


class StreamingFrameService:
    """Publish complete frames during decoding, without a batch-wide reply barrier.

    One bounded decode batch is in flight. Socket handlers only hold the
    condition while updating the cache; neither decoding nor sending holds it.
    The external STAGE1 wire contract and decoder arithmetic remain unchanged.
    """
    def __init__(self, index, cache_dir, decoder, *, maximum_cache_files=32, prefetch_files=25,
                 validate_frame=None):
        import threading
        import time
        if maximum_cache_files < 2 or not 1 <= prefetch_files <= maximum_cache_files:
            raise ValueError('Require 1 <= prefetch_files <= maximum_cache_files and cache >= 2')
        self.index, self.cache_dir, self.decoder = index, str(cache_dir), decoder
        self.maximum_cache_files, self.prefetch_files = maximum_cache_files, prefetch_files
        self._positions = {r.sequence: i for i, r in enumerate(index.frames)}
        self._cache, self._wanted = OrderedDict(), Counter()
        self._condition = threading.Condition()
        self._stop, self._error = False, None
        self._validate = validate_frame or (lambda frame: None)
        self.started_at = time.time()
        self._stats = dict(native_requests=0, native_cache_hits=0, native_batch_decode_calls=0,
                           native_decoded_frames=0, native_evicted_frames=0,
                           native_decode_seconds=0.0, native_bytes_sent=0,
                           native_first_published_seconds=None, native_peak_cache_frames=0)
        self._thread = threading.Thread(target=self._produce, name='kpolaris-ddc-stream', daemon=True)
        self._thread.start()

    def _produce(self):
        import time
        class Stopped(Exception): pass
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or any(s not in self._cache for s in self._wanted))
                    if self._stop: return
                    first = next(s for s in self._wanted if s not in self._cache)
                    position = self._positions[first]
                    sequences = tuple(r.sequence for r in self.index.frames[position:position+self.prefetch_files]
                                      if r.sequence not in self._cache)
                started = time.monotonic()
                published = set()
                def publish(sequence, frame):
                    if sequence not in sequences or sequence in published:
                        raise ValueError('Decoder published an unexpected or duplicate sequence')
                    published.add(sequence)
                    if int(frame.get('sequence', sequence)) != sequence:
                        raise ValueError('Decoder returned a mismatched frame sequence')
                    self._validate(frame)
                    with self._condition:
                        if self._stop: raise Stopped()
                        while len(self._cache) >= self.maximum_cache_files:
                            victim = next((s for s in self._cache if s not in self._wanted), None)
                            if victim is None:
                                self._condition.wait()
                                if self._stop: raise Stopped()
                                continue
                            del self._cache[victim]
                            self._stats['native_evicted_frames'] += 1
                        self._cache[sequence] = frame
                        self._stats['native_decoded_frames'] += 1
                        self._stats['native_peak_cache_frames'] = max(
                            len(self._cache), self._stats['native_peak_cache_frames'])
                        if self._stats['native_first_published_seconds'] is None:
                            self._stats['native_first_published_seconds'] = time.monotonic()-started
                        self._condition.notify_all()
                try:
                    self.decoder(sequences, on_frame=publish)
                    if published != set(sequences):
                        raise RuntimeError('DDC decoder did not publish all requested frames')
                finally:
                    with self._condition:
                        self._stats['native_batch_decode_calls'] += 1
                        self._stats['native_decode_seconds'] += time.monotonic()-started
        except Stopped:
            pass
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def stage_sequence(self, sequence):
        sequence = int(sequence)
        if sequence not in self._positions:
            raise KeyError(f'Sequence {sequence} is not present')
        with self._condition:
            self._stats['native_requests'] += 1
            if sequence in self._cache: self._stats['native_cache_hits'] += 1
            self._wanted[sequence] += 1
            self._condition.notify_all()
            try:
                self._condition.wait_for(lambda: sequence in self._cache or self._error or self._stop)
                if self._error: raise RuntimeError('DDC streaming decode failed') from self._error
                if self._stop: raise RuntimeError('DDC streaming service has stopped')
                self._cache.move_to_end(sequence)
                return self._cache[sequence]
            finally:
                self._wanted[sequence] -= 1
                if not self._wanted[sequence]: del self._wanted[sequence]
                self._condition.notify_all()

    def warm_sequences(self, start_sequence, count):
        if count < 1 or count > self.maximum_cache_files:
            raise ValueError('Warm range must fit the native cache')
        position = self._positions[int(start_sequence)]
        sequences = tuple(r.sequence for r in self.index.frames[position:position+count])
        if len(sequences) != count: raise ValueError('Warm range exceeds index')
        with self._condition:
            before = self._stats['native_decoded_frames']
            self._wanted.update(sequences)
            self._condition.notify_all()
        try:
            for sequence in sequences: self.stage_sequence(sequence)
            return dict(start_sequence=sequences[0], end_sequence=sequences[-1],
                        sequence_count=count, decoded_count=self._stats['native_decoded_frames']-before)
        finally:
            with self._condition:
                self._wanted.subtract(sequences)
                self._wanted += Counter()
                self._condition.notify_all()

    def materialize_sequence(self, sequence):
        raise ValueError('This service supports native STAGE input only; set kharma_ddc_native=1')

    def record_native_bytes_sent(self, count):
        with self._condition: self._stats['native_bytes_sent'] += count

    def statistics(self):
        import time
        with self._condition:
            return dict(format='kpolaris_ddc_service_stats_v1', uptime_seconds=time.time()-self.started_at,
                        cache_dir=self.cache_dir, maximum_cache_files=self.maximum_cache_files,
                        prefetch_files=self.prefetch_files, native_prefetch_mode='streaming_batch',
                        native_reconstruction=getattr(getattr(self.decoder, 'reconstruction', None), 'mode', 'external'),
                        native_cache_frames=len(self._cache), **self._stats,
                        native_mean_decode_seconds_per_frame=self._stats['native_decode_seconds']/max(1,self._stats['native_decoded_frames']),
                        native_error=str(self._error) if self._error else None)

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join()
