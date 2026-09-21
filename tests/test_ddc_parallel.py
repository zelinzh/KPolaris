#!/usr/bin/env python3
"""Exercise bounded codec scheduling, ownership and worker failure propagation."""
import sys
import time
import threading
import os
import tempfile
from collections import OrderedDict
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from kpolaris_ddc_parallel import ParallelSequenceDecoder, StreamingFrameService


class ParallelDecoderTest(unittest.TestCase):
    def decoder(self, fail=False):
        d = ParallelSequenceDecoder.__new__(ParallelSequenceDecoder)
        d.workers, d.cache_chunks, d.scheme = 3, 2, None
        d.manifest_path = Path('manifest.json')
        d.manifest = {}
        d.times = {i: float(i) / 10 for i in range(1, 8)}
        d.keyframes = {}
        state = SimpleNamespace(active=0, maximum=0, open=0)
        lock = threading.Lock()
        class Archive:
            container = {'chunk_frames': 3}
            def __enter__(self):
                self.owner = threading.get_ident()
                with lock: state.open += 1
                return self
            def __exit__(self, *_):
                with lock: state.open -= 1
        class Dataset:
            def __getitem__(self, _):
                return np.arange(24, dtype=np.float32).reshape(2,3,4) + 1
        class Handle:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def __getitem__(self, _): return Dataset()
        def decode(archive, metadata, dataset, i, **kw):
            self.assertEqual(archive.owner, threading.get_ident())
            with lock:
                state.active += 1
                state.maximum = max(state.maximum, state.active)
            try:
                time.sleep(0.002)
                if fail and i == 4: raise ValueError('injected codec failure')
                return kw['start'] + np.float32(i + metadata.get('offset', 0))
            finally:
                with lock: state.active -= 1
        d.sequence = SimpleNamespace(
            select_scheme=lambda *_: ('test', {'archive_paths': ['gop']}),
            group_archive_frames=lambda paths, seqs: {Path('gop'): {i:i for i in seqs}},
            read_archive_metadata_with_retry=lambda _: {'start_file':'start','end_file':'end'},
            phdf_sequence=lambda _: 0,
            existing_raw_keyframe=lambda *args: Path('anchor'),
            open_archive=lambda *args, **kw: Archive())
        d.codec = SimpleNamespace(
            require_h5py=lambda: SimpleNamespace(File=lambda *args: Handle()),
            read_kharma_native_metadata=lambda _: {'num_meshblocks':2},
            transformed=lambda a, d: a,
            frame_tau=lambda a,m,d,i: i/8,
            decode_dataset=decode)
        return d, state

    def test_parallel_output_order_and_ownership(self):
        d, state = self.decoder()
        frames = d((7, 2, 3, 4, 5, 6, 1, 7))
        self.assertEqual(set(frames), set(range(1,8)))
        for i, frame in frames.items():
            self.assertEqual(frame['time'], i/10)
            for value in frame['datasets'].values():
                np.testing.assert_array_equal(value, np.arange(24).reshape(2,3,4)+1+i)
        self.assertGreater(state.maximum, 1)
        self.assertLessEqual(state.maximum, d.workers)
        self.assertEqual((state.open, state.active), (0,0))
        self.assertEqual(d(()), {})
        with self.assertRaises(KeyError): d((999,))

    def test_multiple_archive_contexts_are_not_reused(self):
        d, state = self.decoder()
        d.sequence.group_archive_frames = lambda paths, seqs: {
            Path('first'): {i:i for i in seqs if i<=4},
            Path('second'): {i-4:i for i in seqs if i>4}}
        d.sequence.read_archive_metadata_with_retry = lambda path: {
            'start_file':'start','end_file':'end','offset':4 if path.name=='second' else 0}
        frames = d((7,1,5,4,2,6,3))
        for seq, frame in frames.items():
            for value in frame['datasets'].values():
                np.testing.assert_array_equal(value, np.arange(24).reshape(2,3,4)+1+seq)
        self.assertEqual((state.open,state.active),(0,0))

    def test_keyframes_share_anchor_cache_and_invalidate_on_change(self):
        d, state = self.decoder()
        d._anchors = OrderedDict()
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'anchor.phdf';path.write_bytes(b'first')
            d.keyframes={0:path};d.times[0]=0.
            d.sequence.existing_raw_keyframe=lambda *args:path
            def read(manifest,keys,**kw):
                calls.extend(keys)
                return {seq:{'sequence':seq,'time':d.times[seq],'source_phdf':str(path),
                    'exact_keyframe':True,'par_text':'grid','num_meshblocks':2,
                    'meshblock_size':(2,3,4),'block_order':np.array([[0,0,0],[1,0,0]],dtype='i8'),
                    'datasets':{name:np.arange(24,dtype='f4').reshape(2,3,4) for name in d.datasets}}
                    for seq in keys}
            d.sequence.decode_sequence_frames_to_arrays=read
            first=d((0,))[0];again=d((0,))[0]
            self.assertEqual(calls,[0])
            self.assertEqual(first['source_phdf'],again['source_phdf'])
            self.assertTrue(again['exact_keyframe'])
            for name in d.datasets:
                self.assertIs(first['datasets'][name],again['datasets'][name])
                self.assertFalse(again['datasets'][name].flags.writeable)
            stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1))
            d((0,));self.assertEqual(calls,[0,0])
            for i in range(6):d._remember_anchor((str(i),1,1),{},first['datasets'])
            self.assertLessEqual(len(d._anchors),4)
            # Published arrays remain valid when their cache entries are evicted.
            np.testing.assert_array_equal(first['datasets']['prims.rho'].ravel(),np.arange(24))

    def test_worker_failure_waits_for_cleanup(self):
        d, state = self.decoder(fail=True)
        with self.assertRaisesRegex(ValueError, 'injected codec failure'):
            d(tuple(range(1,8)))
        self.assertEqual((state.open, state.active), (0,0))


class StreamingServiceTest(unittest.TestCase):
    def index(self):
        return SimpleNamespace(frames=[SimpleNamespace(sequence=i) for i in range(12)])

    def test_first_frame_is_returned_before_batch_finishes(self):
        first_sent, release_tail = threading.Event(), threading.Event()
        def decode(sequences, *, on_frame):
            on_frame(sequences[0], {'sequence':sequences[0]})
            first_sent.set()
            if not release_tail.wait(5): raise RuntimeError('test timed out')
            for seq in sequences[1:]: on_frame(seq, {'sequence':seq})
        service=StreamingFrameService(self.index(),'unused',decode,maximum_cache_files=4,prefetch_files=4)
        result=[]
        reader=threading.Thread(target=lambda: result.append(service.stage_sequence(2)))
        try:
            reader.start()
            self.assertTrue(first_sent.wait(2))
            reader.join(1)
            self.assertFalse(reader.is_alive(), 'first frame waited for the entire decode batch')
            self.assertEqual(result,[{'sequence':2}])
            self.assertEqual(service.stage_sequence(2),result[0])
            release_tail.set()
            for seq in range(3,12):self.assertEqual(service.stage_sequence(seq)['sequence'],seq)
            # Backward seeks must re-decode, rather than deadlock or return a wrong frame.
            self.assertEqual(service.stage_sequence(0)['sequence'],0)
            self.assertLessEqual(service.statistics()['native_peak_cache_frames'],4)
        finally:
            release_tail.set();service.close();reader.join(2)
        self.assertFalse(service._thread.is_alive())

    def test_error_and_shutdown_wake_readers(self):
        def fail(sequences, *, on_frame):raise ValueError('injected failure')
        service=StreamingFrameService(self.index(),'unused',fail,maximum_cache_files=2,prefetch_files=2)
        try:
            with self.assertRaisesRegex(RuntimeError,'streaming decode failed'):service.stage_sequence(1)
        finally:service.close()
        with self.assertRaises(RuntimeError):service.stage_sequence(1)

    def test_warm_pins_range_and_missing_frames_fail(self):
        def decode(sequences, *, on_frame):
            for seq in sequences:on_frame(seq,{'sequence':seq})
        service=StreamingFrameService(self.index(),'unused',decode,maximum_cache_files=4,prefetch_files=3)
        try:
            service.warm_sequences(2,4)
            self.assertTrue(all(seq in service._cache for seq in range(2,6)))
            self.assertEqual(service.stage_sequence(11)['sequence'],11)
        finally:service.close()
        service=StreamingFrameService(self.index(),'unused',lambda *a,**k:None,maximum_cache_files=2,prefetch_files=2)
        try:
            with self.assertRaises(RuntimeError):service.stage_sequence(0)
        finally:service.close()


if __name__ == '__main__': unittest.main()
