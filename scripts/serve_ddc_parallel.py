#!/usr/bin/env python3
"""Serve native DDC arrays with bounded parallelism inside each codec archive."""
from __future__ import annotations
import argparse
import json
import os
import signal
import threading
from pathlib import Path
from kpolaris_ddc_parallel import ParallelSequenceDecoder, StreamingFrameService


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codec-project', type=Path, default=os.environ.get('KPOLARIS_DDC_CODEC_PROJECT'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--socket', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--scheme')
    p.add_argument('--maximum-cache-files', type=int, default=32)
    p.add_argument('--prefetch-files', type=int, default=25)
    p.add_argument('--native-decode-workers', type=int, default=8)
    p.add_argument('--native-archive-cache-chunks', type=int, default=8)
    p.add_argument('--native-reconstruction', choices=('auto', 'reference', 'numpy'), default='auto')
    p.add_argument('--native-streaming', type=int, choices=(0, 1), default=1)
    p.add_argument('--ready-file', type=Path)
    p.add_argument('--stats-file', type=Path)
    args = p.parse_args()
    if args.codec_project is None:
        p.error('--codec-project or KPOLARIS_DDC_CODEC_PROJECT is required')
    decoder = ParallelSequenceDecoder(args.codec_project, args.manifest,
        workers=args.native_decode_workers, cache_chunks=args.native_archive_cache_chunks, scheme=args.scheme, reconstruction=args.native_reconstruction)
    from dense_dump_codec.kpolaris_service import DDCFrameService, DDCUnixServer, write_service_statistics, _native_wire_parts
    from dense_dump_codec.sequence import DenseSequenceIndex
    def no_phdf(outputs):
        raise ValueError('This service supports native STAGE input only; set kharma_ddc_native=1')
    index = DenseSequenceIndex.from_manifest(args.manifest)
    if args.native_streaming:
        service = StreamingFrameService(index, args.cache_dir, decoder,
            maximum_cache_files=args.maximum_cache_files, prefetch_files=args.prefetch_files,
            validate_frame=_native_wire_parts)
    else:
        service = DDCFrameService(index, args.cache_dir, no_phdf, decode_staged_frames=decoder,
            maximum_cache_files=args.maximum_cache_files, prefetch_files=args.prefetch_files)
    with DDCUnixServer(args.socket, service) as server:
        def stop(*_):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if args.ready_file:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.write_text(json.dumps({'format': 'kpolaris_ddc_parallel_ready_v1',
                'native_staged_arrays': True, 'streaming': bool(args.native_streaming), 'socket': str(args.socket),
                'decode_workers': args.native_decode_workers,
                'reconstruction': decoder.reconstruction.mode,
                'archive_cache_chunks_per_worker': args.native_archive_cache_chunks}) + '\n')
        try:
            server.serve_forever(poll_interval=0.1)
        finally:
            if args.native_streaming: service.close()
    if args.ready_file:
        args.ready_file.unlink(missing_ok=True)
    if args.stats_file:
        write_service_statistics(args.stats_file, service)
    print(json.dumps(service.statistics(), sort_keys=True))


if __name__ == '__main__':
    main()
