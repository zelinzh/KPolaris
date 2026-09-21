#!/usr/bin/env python3
"""Deterministic mutation fuzzing for the five public GRMHD loaders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time

try:
    import h5py
    import numpy as np
except ImportError:
    print("h5py/numpy is not available; skipping loader fuzz test")
    raise SystemExit(77)

import test_hdf5_contract as contract


SANITIZER_MARKERS = (
    "ERROR: AddressSanitizer",
    "SUMMARY: AddressSanitizer",
    "UndefinedBehaviorSanitizer",
    "runtime error:",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def hdf5_dataset_names(h5: h5py.File) -> list[str]:
    names: list[str] = []

    def collect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    h5.visititems(collect)
    return names


def hdf5_attribute_owners(h5: h5py.File) -> list[str]:
    names = [""] if h5.attrs else []

    def collect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if obj.attrs:
            names.append(name)

    h5.visititems(collect)
    return names


def delete_hdf5_object(h5: h5py.File, name: str) -> None:
    parent_name, _, leaf = name.rpartition("/")
    parent = h5[parent_name] if parent_name else h5
    del parent[leaf]


def mutate_hdf5(
    source: Path, target: Path, case_index: int, rng: random.Random
) -> str:
    shutil.copyfile(source, target)
    mutation = case_index % 7
    if mutation == 0:
        with h5py.File(target, "r+") as h5:
            names = hdf5_dataset_names(h5)
            chosen = rng.choice(names)
            delete_hdf5_object(h5, chosen)
        return f"delete_dataset:{chosen}"
    if mutation == 1:
        with h5py.File(target, "r+") as h5:
            names = [
                name for name in hdf5_dataset_names(h5)
                if h5[name].dtype.kind in "biufc" and h5[name].size
            ]
            chosen = rng.choice(names)
            dataset = h5[chosen]
            numeric_occurrence = (case_index // 7) % 4
            value = (
                [math.nan, math.inf, 0.0, -1.0][numeric_occurrence]
                if dataset.dtype.kind in "fc"
                else [0, -1, 1, 16][numeric_occurrence]
            )
            if dataset.shape:
                index = tuple(0 for _ in dataset.shape)
                dataset[index] = value
            else:
                dataset[()] = value
        return f"replace_numeric:{chosen}:{value}"
    if mutation == 2:
        with h5py.File(target, "r+") as h5:
            candidates = [
                (name, axis)
                for name in hdf5_dataset_names(h5)
                for axis, extent in enumerate(h5[name].shape)
                if extent > 1
            ]
            chosen, axis = rng.choice(candidates)
            data = h5[chosen][...]
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(0, 1)
            shortened = data[tuple(slices)]
            delete_hdf5_object(h5, chosen)
            h5.create_dataset(chosen, data=shortened)
        return (
            f"shorten_dimension:{chosen}:axis={axis}:"
            f"{data.shape}->{shortened.shape}"
        )
    if mutation == 3:
        with h5py.File(target, "r+") as h5:
            owners = hdf5_attribute_owners(h5)
            if not owners:
                names = hdf5_dataset_names(h5)
                chosen = rng.choice(names)
                delete_hdf5_object(h5, chosen)
                return f"delete_dataset_fallback:{chosen}"
            owner_name = rng.choice(owners)
            owner = h5[owner_name] if owner_name else h5
            attribute = rng.choice(list(owner.attrs.keys()))
            del owner.attrs[attribute]
        return f"delete_attribute:{owner_name or '/'}:{attribute}"

    data = bytearray(target.read_bytes())
    if mutation == 4:
        keep = max(1, int(len(data) * (0.15 + 0.7 * rng.random())))
        target.write_bytes(data[:keep])
        return f"truncate_bytes:{len(data)}->{keep}"
    if mutation == 5:
        count = rng.randint(1, 64)
        target.write_bytes(data + bytes(rng.randrange(256) for _ in range(count)))
        return f"append_bytes:{count}"
    start = max(0, len(data) * 3 // 4)
    position = rng.randrange(start, len(data))
    bit = 1 << rng.randrange(8)
    data[position] ^= bit
    target.write_bytes(data)
    return f"flip_tail_bit:{position}:{bit}"


def mutate_athenak(
    source: Path, target: Path, case_index: int, rng: random.Random
) -> str:
    data = bytearray(source.read_bytes())
    mutation = case_index % 5
    if mutation == 0:
        keep = rng.randrange(1, len(data))
        target.write_bytes(data[:keep])
        return f"truncate_bytes:{len(data)}->{keep}"
    if mutation == 1:
        count = rng.randint(1, 64)
        target.write_bytes(data + bytes(rng.randrange(256) for _ in range(count)))
        return f"append_bytes:{count}"
    if mutation == 2:
        position = rng.randrange(max(1, len(data) // 2), len(data))
        bit = 1 << rng.randrange(8)
        data[position] ^= bit
        target.write_bytes(data)
        return f"flip_payload_bit:{position}:{bit}"
    if mutation == 3:
        header_end = data.find(b"<meshblock>")
        candidates = [
            index for index in range(max(header_end, 1))
            if 32 <= data[index] < 127 and data[index] not in b"=0123456789."
        ]
        position = rng.choice(candidates)
        old = data[position]
        data[position] = ord("?") if old != ord("?") else ord("!")
        target.write_bytes(data)
        return f"replace_header_character:{position}:{old}->{data[position]}"
    position = len(data) - rng.randint(1, min(128, len(data)))
    data[position] = 0
    target.write_bytes(data)
    return f"zero_tail_byte:{position}"


def mutate_bhac(
    source: Path, target: Path, case_index: int, rng: random.Random
) -> str:
    data = bytearray(source.read_bytes())
    mutation = case_index % 5
    if mutation == 0:
        keep = rng.randrange(1, len(data))
        target.write_bytes(data[:keep])
        return f"truncate_bytes:{len(data)}->{keep}"
    if mutation == 1:
        count = rng.randint(1, 64)
        target.write_bytes(data + bytes(rng.randrange(256) for _ in range(count)))
        return f"append_bytes:{count}"
    if mutation == 2:
        payload_size = 13 * 8 * 8 * 8 * 8
        position = rng.randrange(payload_size)
        bit = 1 << rng.randrange(8)
        data[position] ^= bit
        target.write_bytes(data)
        return f"flip_payload_bit:{position}:{bit}"
    if mutation == 3:
        header_start = len(data) - struct.calcsize("<8id")
        field = rng.randrange(8)
        value = rng.choice((-1, 0, 2, 16))
        struct.pack_into("<i", data, header_start + 4 * field, value)
        target.write_bytes(data)
        return f"replace_footer_int:{field}:{value}"
    position = len(data) - rng.randint(1, min(88, len(data)))
    data[position] = 0
    target.write_bytes(data)
    return f"zero_footer_byte:{position}"


def mutate_hamr(
    source: Path, target: Path, case_index: int, rng: random.Random
) -> str:
    shutil.copytree(source, target)
    parameters = target / "parameters"
    dump = target / "new_dump0000"
    mutation = case_index % 6
    if mutation == 0:
        data = parameters.read_bytes()
        keep = rng.randrange(1, len(data))
        parameters.write_bytes(data[:keep])
        return f"truncate_parameters:{len(data)}->{keep}"
    if mutation == 1:
        data = dump.read_bytes()
        keep = rng.randrange(1, len(data))
        dump.write_bytes(data[:keep])
        return f"truncate_dump:{len(data)}->{keep}"
    if mutation == 2:
        data = bytearray(dump.read_bytes())
        position = rng.randrange(len(data))
        bit = 1 << rng.randrange(8)
        data[position] ^= bit
        dump.write_bytes(data)
        return f"flip_dump_bit:{position}:{bit}"
    if mutation == 3:
        data = bytearray(parameters.read_bytes())
        offsets = (0x008, 0x00C, 0x040, 0x044, 0x048, 0x04C, 0x050, 0x054, 0x058)
        offset = rng.choice(offsets)
        value = rng.choice((-1, 0, 2, 16))
        struct.pack_into("<i", data, offset, value)
        parameters.write_bytes(data)
        return f"replace_parameter_int:{offset:#x}:{value}"
    if mutation == 4:
        count = rng.randint(1, 64)
        dump.write_bytes(
            dump.read_bytes() + bytes(rng.randrange(256) for _ in range(count)))
        return f"append_dump_bytes:{count}"
    extra = target / f"new_dump{rng.randint(1, 9999):04d}"
    extra.write_bytes(bytes(rng.randrange(256) for _ in range(rng.randint(1, 64))))
    return f"add_dump_file:{extra.name}:{extra.stat().st_size}"


def validate_successful_image(path: Path) -> None:
    if not path.is_file():
        raise AssertionError(f"successful loader did not write {path}")
    with h5py.File(path, "r") as h5:
        if contract.text_attr(h5.attrs["schema"]) != "kpolaris_image":
            raise AssertionError(f"unexpected output schema in {path}")
        for stokes in "IQUV":
            values = h5[f"frame_0/freq_0/{stokes}"][...]
            if not np.isfinite(values).all():
                raise AssertionError(
                    f"successful mutated input produced nonfinite {stokes}: {path}")
        if "frame_0/diagnostics/reason" not in h5:
            raise AssertionError(f"missing termination diagnostics in {path}")


def run_case(
    model: str,
    executable: Path,
    transport_args: list[str],
    output: Path,
    case_dir: Path,
    mutation: str,
    timeout: float,
) -> dict[str, object]:
    command = [
        str(executable),
        *transport_args,
        "--format=hdf5",
        f"--output={output}",
        "--parameter_output=none",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"{model} mutation timed out after {timeout}s: {mutation}") from error
    elapsed = time.perf_counter() - started
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    combined = completed.stdout + completed.stderr
    if completed.returncode < 0:
        raise AssertionError(
            f"{model} mutation terminated by signal {-completed.returncode}: {mutation}")
    markers = [marker for marker in SANITIZER_MARKERS if marker in combined]
    if markers:
        raise AssertionError(
            f"{model} mutation triggered sanitizer {markers}: {mutation}")
    if completed.returncode == 0:
        validate_successful_image(output)
        outcome = "accepted"
    else:
        if not combined.strip():
            raise AssertionError(
                f"{model} mutation failed without a diagnostic: {mutation}")
        outcome = "rejected"
    return {
        "model": model,
        "mutation": mutation,
        "outcome": outcome,
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "stdout_sha256": sha256(case_dir / "stdout.log"),
        "stderr_sha256": sha256(case_dir / "stderr.log"),
        "output": str(output) if output.exists() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--iharm-image-exe", type=Path)
    parser.add_argument("--kharma-image-exe", type=Path)
    parser.add_argument("--athenak-image-exe", type=Path)
    parser.add_argument("--bhac-image-exe", type=Path)
    parser.add_argument("--hamr-image-exe", type=Path)
    parser.add_argument("--cases-per-loader", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.cases_per_loader < 1:
        parser.error("--cases-per-loader must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    executables = {
        name: path.resolve()
        for name, path in (
            ("iharm", args.iharm_image_exe),
            ("kharma", args.kharma_image_exe),
            ("athenak", args.athenak_image_exe),
            ("bhac", args.bhac_image_exe),
            ("hamr", args.hamr_image_exe),
        )
        if path is not None
    }
    if not executables:
        print("no GRMHD loader executable is available; skipping loader fuzz test")
        return 77
    for executable in executables.values():
        if not executable.is_file():
            raise FileNotFoundError(executable)

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    corpus = workdir / "corpus"
    corpus.mkdir()
    bases: dict[str, Path] = {}
    if "iharm" in executables:
        bases["iharm"] = corpus / "iharm.h5"
        contract.write_synthetic_iharm_fixture(bases["iharm"])
    if "kharma" in executables:
        bases["kharma"] = corpus / "kharma.phdf"
        contract.write_synthetic_kharma_fixture(bases["kharma"])
    if "athenak" in executables:
        bases["athenak"] = corpus / "athenak.bin"
        contract.write_synthetic_athenak_fixture(bases["athenak"])
    if "bhac" in executables:
        bases["bhac"] = corpus / "bhac.dat"
        contract.write_synthetic_bhac_fixture(bases["bhac"])
    if "hamr" in executables:
        bases["hamr"] = corpus / "hamr"
        contract.write_synthetic_hamr_fixture(bases["hamr"])

    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []
    transport_builders = {
        "iharm": contract.synthetic_iharm_transport_args,
        "kharma": contract.synthetic_kharma_transport_args,
        "athenak": contract.synthetic_athenak_transport_args,
        "bhac": contract.synthetic_bhac_transport_args,
        "hamr": contract.synthetic_hamr_transport_args,
    }
    for model, executable in executables.items():
        model_dir = workdir / model
        model_dir.mkdir()
        for case_index in range(args.cases_per_loader):
            case_dir = model_dir / f"case_{case_index:03d}"
            case_dir.mkdir()
            base = bases[model]
            if model in ("iharm", "kharma"):
                mutated = case_dir / base.name
                mutation = mutate_hdf5(base, mutated, case_index, rng)
            elif model == "athenak":
                mutated = case_dir / base.name
                mutation = mutate_athenak(base, mutated, case_index, rng)
            elif model == "bhac":
                mutated = case_dir / base.name
                mutation = mutate_bhac(base, mutated, case_index, rng)
            else:
                mutated = case_dir / "hamr"
                mutation = mutate_hamr(base, mutated, case_index, rng)
            output = case_dir / "output.h5"
            record = run_case(
                model,
                executable,
                transport_builders[model](mutated),
                output,
                case_dir,
                mutation,
                args.timeout,
            )
            record["case"] = case_index
            records.append(record)
            print(
                f"{model} case {case_index + 1}/{args.cases_per_loader}: "
                f"{record['outcome']} ({mutation})",
                flush=True,
            )

    accepted = sum(record["outcome"] == "accepted" for record in records)
    rejected = len(records) - accepted
    signature_payload = [
        {
            key: record[key]
            for key in ("model", "case", "mutation", "outcome", "returncode")
        }
        for record in records
    ]
    signature = hashlib.sha256(
        json.dumps(
            signature_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    harness = Path(__file__).resolve()
    summary = {
        "schema": "kpolaris_loader_mutation_fuzz",
        "schema_version": 1,
        "seed": args.seed,
        "cases_per_loader": args.cases_per_loader,
        "timeout_s": args.timeout,
        "total_cases": len(records),
        "accepted": accepted,
        "rejected": rejected,
        "mutation_signature_sha256": signature,
        "models": sorted(executables),
        "harness": {"path": str(harness), **file_record(harness)},
        "executables": {
            model: {"path": str(path), **file_record(path)}
            for model, path in executables.items()
        },
        "corpus": {
            model: (
                {str(path.relative_to(workdir)): file_record(path)}
                if path.is_file()
                else {
                    str(item.relative_to(workdir)): file_record(item)
                    for item in sorted(path.rglob("*")) if item.is_file()
                }
            )
            for model, path in bases.items()
        },
        "records": records,
        "claim_boundary": (
            "Deterministic structured mutation coverage; this is not an "
            "exhaustive proof of parser safety for arbitrary hostile files."
        ),
    }
    summary_path = workdir / "fuzz-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"loader_fuzz total={len(records)} accepted={accepted} "
        f"rejected={rejected} seed={args.seed}"
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
