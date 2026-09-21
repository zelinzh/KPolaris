#!/usr/bin/env python3
"""Build a bounded, lossless contiguous HDF5 working cache for KHARMA dumps.

Only prims.rho/u/uvec/B storage layout changes. All other HDF5 objects are
copied; original files are never modified. Requires h5py and numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import h5py

FIELDS = ("prims.rho", "prims.u", "prims.uvec", "prims.B")


def cache_dump(source: Path, target: Path, remaining_bytes: int) -> dict:
    if target.exists():
        raise FileExistsError(f"refusing to replace existing cache: {target}")
    with h5py.File(source, "r") as src:
        for field in FIELDS:
            if field not in src or src[field].dtype.kind != "f":
                raise ValueError(f"{source}: missing floating-point dataset {field}")
        payload_bytes = sum(src[field].size * src[field].dtype.itemsize for field in FIELDS)
        if payload_bytes > remaining_bytes:
            raise ValueError(f"cache budget insufficient for {source.name}: primitives need {payload_bytes} bytes")
        descriptor, temporary = tempfile.mkstemp(prefix=".kharma-cache-", suffix=".h5", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            hashes = {}
            with h5py.File(temporary, "w") as dst:
                dst.attrs.update(src.attrs)
                for name in src:
                    if name not in FIELDS:
                        src.copy(name, dst)
                        continue
                    values = src[name][...]
                    dataset = dst.create_dataset(name, data=values, dtype=src[name].dtype)
                    dataset.attrs.update(src[name].attrs)
                    hashes[name] = hashlib.sha256(values.tobytes()).hexdigest()
            size = temporary.stat().st_size
            if size > remaining_bytes:
                raise ValueError(f"cache budget insufficient including metadata: {size} bytes")
            # Verify bytes, including signed zeros and NaN payloads, after writing.
            with h5py.File(temporary, "r") as cached:
                for field in FIELDS:
                    a, b = src[field], cached[field]
                    if a.shape != b.shape or a.dtype != b.dtype:
                        raise ValueError(f"cache shape/dtype mismatch: {field}")
                    if hashlib.sha256(b[...].tobytes()).hexdigest() != hashes[field]:
                        raise ValueError(f"cache primitive mismatch: {field}")
            # Linking refuses a racing pre-existing target, unlike os.replace.
            os.link(temporary, target)
            return {
                "source": str(source), "cache": str(target),
                "source_bytes": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
                "cache_bytes": size,
                "primitive_sha256": hashes, "bitwise_verified": True,
            }
        finally:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-cache-gib", type=float, default=4.0,
                        help="maximum cache payload including existing files; manifest excluded (default 4 GiB)")
    args = parser.parse_args()
    if not math.isfinite(args.max_cache_gib) or args.max_cache_gib <= 0:
        parser.error("--max-cache-gib must be finite and positive")
    sources = [p.resolve(strict=True) for p in args.inputs]
    if len({p.name for p in sources}) != len(sources):
        parser.error("input basenames must be unique")
    if any(p.name == "cache_manifest.json" for p in sources):
        parser.error("cache_manifest.json is reserved for provenance")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "cache_manifest.json"
    if manifest.exists() or any((output / p.name).exists() for p in sources):
        parser.error("destination already contains a manifest or an input filename; use a fresh directory")
    budget = int(args.max_cache_gib * 1024**3)
    used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    records = []
    payload = {"schema": "kpolaris_kharma_contiguous_cache_v1", "max_bytes": budget,
               "complete": False, "records": records}
    try:
        for source in sources:
            record = cache_dump(source, output / source.name, budget - used)
            used += record["cache_bytes"]
            records.append(record)
            print(f"cached {source.name}: {record['cache_bytes']} bytes; primitives verified", flush=True)
        payload["complete"] = True
    finally:
        # Retain provenance even when a later input exceeds the budget.
        # The JSON manifest itself is small and excluded from the payload budget.
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
