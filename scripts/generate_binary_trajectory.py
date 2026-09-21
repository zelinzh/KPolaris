#!/usr/bin/env python3
"""Generate a KPolaris SKS trajectory with the Combi--Ressler CBwaves workflow.

The upstream source is supplied by the user at run time.  This script verifies
the v1 Zenodo source, copies it to a temporary directory, applies the published
4PN denominator correction there, builds CBwaves, and converts its text output
to a self-describing ``G=c=M_ref=1`` HDF5 table.

Upstream implementation and citation:
  https://doi.org/10.5281/zenodo.10841021 (CC BY 4.0 record)
  https://arxiv.org/abs/2403.13308

No upstream source is vendored into KPolaris.  The archive contains third-party
components with their own notices, so redistributing a copied source tree
requires a separate license review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import zipfile


SCHEMA_NAME = "kpolaris.binary_trajectory.v1"
SCHEMA_VERSION = 1
GENERATOR_VERSION = "1.3"
REMNANT_FIT_VERSION = "kpolaris-paper-nr-fit-v1"
MERGER_REACH_ABSOLUTE_TOLERANCE_M = 1.0e-10
MERGER_REACH_CONTRACT = "selected_merger_state_fixed_tolerance_v1"
PRECESSION_REFERENCE_SHA256 = (
    "e9ba566dc7e7b81714f5ebe0949dce8eaf64dc7b6798d062537d4feee6e545b4"
)
SPEED_OF_LIGHT_KM_S = 299792.458

ZENODO_DOI = "10.5281/zenodo.10841021"
ZENODO_ARCHIVE_SHA256 = (
    "3dbadbd5af50355a71378f2cb76053da61f453e66bb61696334ae394909b1f42"
)
UPSTREAM_FILE_SHA256 = {
    "src/cbwaves.cxx": (
        "4a094a9bbac3b2bc142015b70afd5dc939898b7b38a34b05917dd94e94829a9a"
    ),
    "src/ConfigFile.cxx": (
        "c524d1c83c69f879a28349c0e524c068094f6770541d91e57b19eab3536efd0f"
    ),
    "src/precession.py": PRECESSION_REFERENCE_SHA256,
    "include/ConfigFile.h": (
        "05415148de05f93c40e52e4ea67ba8e1c31151e32c0a249fe1db13e38f880741"
    ),
    "include/RK4.h": (
        "f5421a36aa05d4fa3ed13b9b4eb8042c25529731847261bce750a9fd5b094c40"
    ),
    "include/Spline3.h": (
        "39f2bbee3315eb26cb3f7e6bcff4d25d6f2854437d1c7343435dcc31fa6295bb"
    ),
    "include/fft.h": (
        "b4f40f6546224637cd248fcaf19b66e0f8a5ce50eda07f06487cb9faa882142f"
    ),
    "include/hmodes.h": (
        "396cd117c5d75b45ec4a3d628fa18d24feb483d1ebe918c6134ea0544cc0a57b"
    ),
    "include/tvalarray.h": (
        "fff28ce5858e21b201dafff70ebe9034fc03cc0a064f484db7e2ed43659bfd42"
    ),
    "include/tvector.h": (
        "4c4c1d955bd7a5328b8288917f83ab5190cce56b580207cd46eaae09f29499ed"
    ),
    "CMakeLists.txt": (
        "852cbe68675f567d07c3eaeca4a8dad983a0d48fd057e322e1236f22d904252a"
    ),
}
UPSTREAM_CBWAVES_SHA256 = UPSTREAM_FILE_SHA256["src/cbwaves.cxx"]
PATCHED_CBWAVES_SHA256 = (
    "0a28a193f0964164876968b159b9df0c261190f4a605de138592878e967c6067"
)
FOUR_PN_BAD = b"9130111./3306."
FOUR_PN_FIXED = b"9130111./3360."

# These are deliberately the exact constants used by the archived wrapper.
CBWAVES_C_M_PER_S = 2.99792458e8
CBWAVES_MSUN_LENGTH_M = 1476.62504
CBWAVES_MSUN_TIME_S = CBWAVES_MSUN_LENGTH_M / CBWAVES_C_M_PER_S

PN_TERMS = (
    "PN",
    "2PN",
    "SO",
    "SS",
    "RR",
    "PNSO",
    "3PN",
    "1RR",
    "2PNSO",
    "RRSO",
    "RRSS",
    "4PN",
)
DAT_COLUMNS = (
    "t",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "vx",
    "vy",
    "vz",
    "s1x",
    "s1y",
    "s1z",
    "s2x",
    "s2y",
    "s2z",
    "Lx",
    "Ly",
    "Lz",
    "r",
)
OFFICIAL_WRAPPER_DAT_COLUMNS = (
    "t",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "s1x",
    "s1y",
    "s1z",
    "s2x",
    "s2y",
    "s2z",
    "Lx",
    "Ly",
    "Lz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def locate_cbwaves_tree(path: Path) -> Path:
    candidates = (path, path / "CBwaves", path / "public_repo" / "CBwaves")
    for candidate in candidates:
        if (candidate / "src" / "cbwaves.cxx").is_file():
            return candidate.resolve()
    raise ValueError(
        f"cannot locate CBwaves below {path}; expected src/cbwaves.cxx, "
        "CBwaves/src/cbwaves.cxx, or public_repo/CBwaves/src/cbwaves.cxx"
    )


def verify_cbwaves_tree(root: Path) -> dict[str, str]:
    root = locate_cbwaves_tree(root)
    observed: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        actual = sha256_file(path)
        observed[relative] = actual
        if actual != expected:
            errors.append(f"{relative}: expected {expected}, got {actual}")
    if errors:
        raise ValueError(
            "CBwaves source is not the verified Zenodo v1/author source:\n  "
            + "\n  ".join(errors)
        )
    source = root / "src" / "cbwaves.cxx"
    if source.read_bytes().count(FOUR_PN_BAD) != 1:
        raise ValueError("verified cbwaves.cxx does not contain exactly one 3306 typo")
    return observed


def verify_zip_member(member: zipfile.ZipInfo) -> None:
    member_path = Path(member.filename)
    mode = member.external_attr >> 16
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"unsafe path in CBwaves archive: {member.filename!r}")
    if stat.S_ISLNK(mode):
        raise ValueError(f"symbolic link is not accepted in CBwaves archive: {member.filename!r}")


def extract_verified_archive(archive: Path, destination: Path) -> Path:
    actual = sha256_file(archive)
    if actual != ZENODO_ARCHIVE_SHA256:
        raise ValueError(
            "CBwaves archive hash mismatch: expected Zenodo v1 "
            f"{ZENODO_ARCHIVE_SHA256}, got {actual}"
        )
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            verify_zip_member(member)
        bundle.extractall(destination)
    return locate_cbwaves_tree(destination)


def prepare_patched_source(source_input: Path, work_root: Path) -> tuple[Path, dict[str, str]]:
    if source_input.is_file():
        if not zipfile.is_zipfile(source_input):
            raise ValueError("--cbwaves-source file must be the official public_repo.zip")
        extracted = work_root / "archive"
        extracted.mkdir()
        source_root = extract_verified_archive(source_input, extracted)
        archive_hash = ZENODO_ARCHIVE_SHA256
    elif source_input.is_dir():
        source_root = locate_cbwaves_tree(source_input)
        archive_hash = "not-applicable-directory-input"
    else:
        raise ValueError(f"CBwaves source does not exist: {source_input}")

    hashes = verify_cbwaves_tree(source_root)
    copied = work_root / "cbwaves-patched"
    shutil.copytree(source_root, copied)
    source_path = copied / "src" / "cbwaves.cxx"
    contents = source_path.read_bytes()
    if contents.count(FOUR_PN_BAD) != 1 or FOUR_PN_FIXED in contents:
        raise ValueError("refusing ambiguous 4PN patch")
    source_path.write_bytes(contents.replace(FOUR_PN_BAD, FOUR_PN_FIXED, 1))
    patched_hash = sha256_file(source_path)
    if patched_hash != PATCHED_CBWAVES_SHA256:
        raise ValueError(
            f"patched cbwaves.cxx hash mismatch: expected {PATCHED_CBWAVES_SHA256}, "
            f"got {patched_hash}"
        )
    provenance = {
        "source_verification": "verified-zenodo-record-10841021",
        "archive_sha256": archive_hash,
        "upstream_cbwaves_sha256": hashes["src/cbwaves.cxx"],
        "patched_cbwaves_sha256": patched_hash,
        "verified_source_files_sha256_json": canonical_json(hashes),
        "patch": "9130111./3306. -> 9130111./3360. (one occurrence)",
    }
    return copied, provenance


def run_checked(
    command: list[str], *, cwd: Path, timeout_seconds: float | None = None
) -> None:
    print("+ " + " ".join(command), file=sys.stderr)
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        timeout=timeout_seconds,
    )


def build_cbwaves(
    source_root: Path,
    work_root: Path,
    cmake: str,
    jobs: int,
    boost_root: Path | None,
    cmake_prefix_paths: list[Path],
) -> Path:
    build_root = work_root / "build"
    configure = [
        cmake,
        "-S",
        str(source_root),
        "-B",
        str(build_root),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if boost_root is not None:
        configure.append(f"-DBOOST_ROOT={boost_root.resolve()}")
    if cmake_prefix_paths:
        configure.append(
            "-DCMAKE_PREFIX_PATH="
            + ";".join(str(path.resolve()) for path in cmake_prefix_paths)
        )
    run_checked(configure, cwd=work_root)
    run_checked(
        [cmake, "--build", str(build_root), "--parallel", str(jobs)],
        cwd=work_root,
    )
    for name in ("CBwaves", "CBwaves.exe"):
        binary = build_root / name
        if binary.is_file():
            return binary
    raise ValueError(f"CBwaves build succeeded but no executable was found in {build_root}")


def rr_parameters(mass1: float, mass2: float) -> dict[str, float]:
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    return {
        "alpha": 4.0,
        "beta": 5.0,
        "delta1": -99.0 / 14.0 + 27.0 * eta,
        "delta2": 5.0 * (1.0 - 4.0 * eta),
        "delta3": 274.0 / 7.0 + 67.0 / 21.0 * eta,
        "delta4": 2.5 * (1.0 - eta),
        "delta5": -(292.0 + 57.0 * eta) / 7.0,
        "delta6": 51.0 / 28.0 + 71.0 / 14.0 * eta,
    }


def render_cbwaves_ini(args: argparse.Namespace, outfile: str = "trajectory.dat") -> str:
    total_solar = args.reference_mass_solar
    mass_length = total_solar * CBWAVES_MSUN_LENGTH_M
    mass_time = total_solar * CBWAVES_MSUN_TIME_S
    period_m = 2.0 * math.pi * args.initial_separation_M ** 1.5
    period_s = period_m * mass_time
    correction_mask = ",".join(f"'{term}'" for term in PN_TERMS)
    output_columns = ",".join(DAT_COLUMNS)
    rr = rr_parameters(args.mass1_fraction, args.mass2_fraction)
    values: list[tuple[str, object]] = [
        ("m1", args.mass1_fraction * mass_length),
        ("m2", args.mass2_fraction * mass_length),
        ("tmax", args.tmax_M * mass_time),
        ("orbitsmax", args.orbits_max),
        ("T", period_s),
        ("f", 1.0 / period_s),
        ("dt", args.dt_M * mass_time),
        ("epsilon", args.eccentricity),
        ("rmin", args.rmin_M * mass_length),
        ("rmax", args.rmax_M * mass_length),
        ("r", args.initial_separation_M * mass_length),
        ("D", 1.0e9 * mass_length),
        ("iota", 90.0),
        ("phi", 0.0),
        ("theta", 0.0),
        ("varphi", 0.0),
        ("psi", 0.0),
        ("s1x", args.spin1[0]),
        ("s1y", args.spin1[1]),
        ("s1z", args.spin1[2]),
        ("s2x", args.spin2[0]),
        ("s2y", args.spin2[1]),
        ("s2z", args.spin2[2]),
        ("hterms", "'Q'"),
        ("corrs", correction_mask),
        # Kept for compatibility with the archived Python wrapper.  The v1
        # C++ source does not read this key and always performs its built-in
        # initial-velocity/eccentricity loop.
        ("eccapprox", "yes"),
        # The archived checkpoint reader has an out-of-bounds name table.
        ("checkpoint", "no"),
        ("description", "KPolaris patched Combi-Ressler paper trajectory"),
        ("printstep", args.print_step),
        ("printorbit", 0),
        ("loglevel", args.log_level),
        ("adaptive", "no"),
        ("adaptive_step", 1000),
    ]
    values.extend(rr.items())
    lines = [
        "[output]",
        f"outfile = {outfile}",
        "ftfile = trajectory.ft",
        f"outvars = {output_columns}",
        "",
        "[input]",
    ]
    for key, value in values:
        if isinstance(value, float):
            rendered = f"{value:.17g}"
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def _require_numpy_h5py() -> tuple[Any, Any]:
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "trajectory conversion requires NumPy and h5py"
        ) from error
    return np, h5py


def parse_columns(text: str) -> tuple[str, ...]:
    columns = tuple(item.strip() for item in text.split(",") if item.strip())
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("--dat-columns must contain a nonempty list of unique names")
    return columns


def resolve_dat_columns(path: Path, specification: str) -> tuple[str, ...]:
    if specification.strip().lower() != "auto":
        return parse_columns(specification)
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            first = next(
                line for line in stream if line.strip() and not line.lstrip().startswith("#")
            )
    except (OSError, StopIteration) as error:
        raise ValueError(f"cannot inspect CBwaves DAT columns in {path}") from error
    count = len(first.split())
    if count == len(DAT_COLUMNS):
        return DAT_COLUMNS
    if count == len(OFFICIAL_WRAPPER_DAT_COLUMNS):
        return OFFICIAL_WRAPPER_DAT_COLUMNS
    raise ValueError(
        f"cannot infer names for {count} DAT columns; pass --dat-columns explicitly"
    )


def load_cbwaves_dat(
    path: Path,
    *,
    columns: tuple[str, ...],
    total_mass_solar: float,
    mass1: float,
    mass2: float,
) -> dict[str, Any]:
    np, _ = _require_numpy_h5py()
    try:
        table = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read CBwaves DAT file {path}: {error}") from error
    if table.ndim != 2 or table.shape[1] != len(columns):
        raise ValueError(
            f"DAT column mismatch: file has shape {table.shape}, but "
            f"--dat-columns names {len(columns)} columns"
        )
    if table.shape[0] < 3:
        raise ValueError("CBwaves DAT file must contain at least three rows")
    if not np.all(np.isfinite(table)):
        raise ValueError("CBwaves DAT file contains NaN or infinite values")
    index = {name: number for number, name in enumerate(columns)}
    required = {
        "t",
        "x1",
        "y1",
        "z1",
        "x2",
        "y2",
        "z2",
        "s1x",
        "s1y",
        "s1z",
        "s2x",
        "s2y",
        "s2z",
    }
    missing = sorted(required - index.keys())
    if missing:
        raise ValueError(f"DAT file lacks required columns: {', '.join(missing)}")

    length_scale = total_mass_solar * CBWAVES_MSUN_LENGTH_M
    time_scale = total_mass_solar * CBWAVES_MSUN_TIME_S
    time = table[:, index["t"]] / time_scale
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("CBwaves time samples must be finite and strictly increasing")
    position = np.empty((len(time), 2, 3), dtype=np.float64)
    spin_chi = np.empty_like(position)
    for hole in range(2):
        label = hole + 1
        for axis, name in enumerate("xyz"):
            position[:, hole, axis] = table[:, index[f"{name}{label}"]] / length_scale
            spin_chi[:, hole, axis] = table[:, index[f"s{label}{name}"]]

    if {"vx", "vy", "vz"} <= index.keys():
        relative_velocity = np.column_stack(
            [table[:, index[f"v{name}"]] for name in "xyz"]
        )
        velocity = np.empty_like(position)
        velocity[:, 0, :] = mass2 * relative_velocity
        velocity[:, 1, :] = -mass1 * relative_velocity
        velocity_source = "CBwaves relative vx/vy/vz with Newtonian COM split"
    else:
        velocity = np.gradient(position, time, axis=0, edge_order=2)
        velocity_source = "second-order numerical derivative of x1/x2"

    if np.max(np.linalg.norm(velocity, axis=2)) >= 1.0:
        raise ValueError("CBwaves trajectory contains a superluminal individual velocity")
    if np.max(np.linalg.norm(spin_chi, axis=2)) > 1.0 + 1.0e-10:
        raise ValueError("CBwaves inspiral contains |spin_chi| > 1")
    separation = np.linalg.norm(position[:, 0, :] - position[:, 1, :], axis=1)
    result: dict[str, Any] = {
        "time": time,
        "position": position,
        "velocity": velocity,
        "spin_chi": spin_chi,
        "separation": separation,
        "velocity_source": velocity_source,
        "dat_sha256": sha256_file(path),
    }
    if {"Lx", "Ly", "Lz"} <= index.keys():
        # Only its direction is subsequently useful; leave upstream magnitude
        # untouched and label it explicitly in the file.
        result["orbital_angular_momentum"] = np.column_stack(
            [table[:, index[f"L{name}"]] for name in "xyz"]
        )
    return result


def smooth_window(time: Any, start: float, end: float) -> Any:
    np, _ = _require_numpy_h5py()
    values = np.asarray(time, dtype=np.float64)
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("transition start/end must be finite and strictly ordered")
    result = np.zeros_like(values)
    result[values >= end] = 1.0
    interior = (values > start) & (values < end)
    u = (values[interior] - start) / (end - start)
    log_b_over_a = -1.0 / (1.0 - u) + 1.0 / u
    # a/(a+b) evaluated as a stable logistic without underflowing both a,b.
    result[interior] = 1.0 / (1.0 + np.exp(np.clip(log_b_over_a, -745.0, 709.0)))
    # Preserve the mathematical open interval.  Otherwise floating-point
    # rounding can create W==1 before ``end`` and make the native reader treat
    # that sample as the exact-remnant endpoint.
    result[interior] = np.clip(
        result[interior],
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )
    return result


def smooth_window_derivative(time: Any, start: float, end: float) -> Any:
    """Return dW/dt, including its exact zero endpoint limits."""
    np, _ = _require_numpy_h5py()
    values = np.asarray(time, dtype=np.float64)
    window = smooth_window(values, start, end)
    derivative = np.zeros_like(values)
    interior = (values > start) & (values < end)
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    # Once the represented W has saturated at the nearest interior float it is
    # locally a plateau.  Treating the clipped value as the exact exponential
    # in the analytic formula could multiply epsilon by an enormous 1/(1-u)^2.
    active = interior & (window > lower) & (window < upper)
    u = (values[active] - start) / (end - start)
    derivative[active] = (
        window[active]
        * (1.0 - window[active])
        * (1.0 / u**2 + 1.0 / (1.0 - u) ** 2)
        / (end - start)
    )
    return derivative


def interpolate_components(time: Any, values: Any, query: Any) -> Any:
    np, _ = _require_numpy_h5py()
    source_time = np.asarray(time, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    target = np.asarray(query, dtype=np.float64)
    flat = source.reshape((source.shape[0], -1))
    interpolated = np.column_stack(
        [np.interp(target, source_time, flat[:, component]) for component in range(flat.shape[1])]
    )
    return interpolated.reshape((len(target),) + source.shape[1:])


def interpolate_cubic_hermite(
    time: Any, values: Any, derivatives: Any, query: Any
) -> tuple[Any, Any]:
    """C1 interpolation whose returned derivative is analytically consistent."""
    np, _ = _require_numpy_h5py()
    source_time = np.asarray(time, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    source_derivative = np.asarray(derivatives, dtype=np.float64)
    target = np.asarray(query, dtype=np.float64)
    if source.shape != source_derivative.shape or source.shape[0] != len(source_time):
        raise ValueError("Hermite position and velocity arrays have inconsistent shapes")
    if np.any(target < source_time[0]) or np.any(target > source_time[-1]):
        raise ValueError("Hermite interpolation query lies outside the source table")
    interval = np.searchsorted(source_time, target, side="right") - 1
    interval = np.clip(interval, 0, len(source_time) - 2)
    t0 = source_time[interval]
    step = source_time[interval + 1] - t0
    u = (target - t0) / step
    expand = (slice(None),) + (None,) * (source.ndim - 1)
    u_e = u[expand]
    step_e = step[expand]
    y0 = source[interval]
    y1 = source[interval + 1]
    d0 = source_derivative[interval]
    d1 = source_derivative[interval + 1]
    h00 = 2.0 * u_e**3 - 3.0 * u_e**2 + 1.0
    h10 = u_e**3 - 2.0 * u_e**2 + u_e
    h01 = -2.0 * u_e**3 + 3.0 * u_e**2
    h11 = u_e**3 - u_e**2
    interpolated = h00 * y0 + h10 * step_e * d0 + h01 * y1 + h11 * step_e * d1
    dh00 = 6.0 * u_e**2 - 6.0 * u_e
    dh10 = 3.0 * u_e**2 - 4.0 * u_e + 1.0
    dh01 = -dh00
    dh11 = 3.0 * u_e**2 - 2.0 * u_e
    derivative = (
        dh00 * y0 + dh10 * step_e * d0 + dh01 * y1 + dh11 * step_e * d1
    ) / step_e
    return interpolated, derivative


def _unit_vector(vector: Any, label: str) -> Any:
    np, _ = _require_numpy_h5py()
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (3,) or not np.all(np.isfinite(values)) or norm <= 1.0e-14:
        raise ValueError(f"cannot construct {label} from a finite nonzero 3-vector")
    return values / norm


def _spin_angles(spin1: Any, spin2: Any, orbital_axis: Any) -> tuple[float, float, float]:
    """Return theta1, theta2 and the signed in-plane Delta-phi."""
    np, _ = _require_numpy_h5py()
    axis = _unit_vector(orbital_axis, "orbital angular-momentum direction")
    spins = (np.asarray(spin1, dtype=np.float64), np.asarray(spin2, dtype=np.float64))
    theta: list[float] = []
    projected: list[Any] = []
    for spin in spins:
        magnitude = float(np.linalg.norm(spin))
        if magnitude <= 1.0e-15:
            theta.append(0.0)
            projected.append(np.zeros(3, dtype=np.float64))
            continue
        cosine = float(np.clip(np.dot(spin, axis) / magnitude, -1.0, 1.0))
        theta.append(math.acos(cosine))
        projected.append(spin - np.dot(spin, axis) * axis)
    norm1 = float(np.linalg.norm(projected[0]))
    norm2 = float(np.linalg.norm(projected[1]))
    if norm1 <= 1.0e-15 or norm2 <= 1.0e-15:
        delta_phi = 0.0
    else:
        p1 = projected[0] / norm1
        p2 = projected[1] / norm2
        delta_phi = math.atan2(
            float(np.dot(axis, np.cross(p1, p2))),
            float(np.clip(np.dot(p1, p2), -1.0, 1.0)),
        )
    return theta[0], theta[1], delta_phi


def paper_nr_final_mass(
    q: float, spin1: Any, spin2: Any, orbital_axis: Any
) -> float:
    """Barausse--Morozova--Rezzolla 2012 Eq. (18), with M_initial=1."""
    np, _ = _require_numpy_h5py()
    axis = _unit_vector(orbital_axis, "orbital angular-momentum direction")
    spin1 = np.asarray(spin1, dtype=np.float64)
    spin2 = np.asarray(spin2, dtype=np.float64)
    eta = q / (1.0 + q) ** 2
    chi_tilde_parallel = (
        float(np.dot(spin1, axis)) + q**2 * float(np.dot(spin2, axis))
    ) / (1.0 + q) ** 2
    if abs(chi_tilde_parallel) > 1.0 + 1.0e-12:
        raise ValueError("BMR12 effective aligned spin lies outside [-1,1]")
    chi_tilde_parallel = float(np.clip(chi_tilde_parallel, -1.0, 1.0))
    z1 = 1.0 + np.cbrt(1.0 - chi_tilde_parallel**2) * (
        np.cbrt(1.0 + chi_tilde_parallel)
        + np.cbrt(1.0 - chi_tilde_parallel)
    )
    z2 = math.sqrt(3.0 * chi_tilde_parallel**2 + z1**2)
    spin_sign = 0.0 if chi_tilde_parallel == 0.0 else math.copysign(1.0, chi_tilde_parallel)
    risco = 3.0 + z2 - spin_sign * math.sqrt(
        max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    )
    eisco = math.sqrt(1.0 - 2.0 / (3.0 * risco))
    p0 = 0.04827
    p1 = 0.01707
    radiated = eta * (1.0 - eisco) + 4.0 * eta**2 * (
        4.0 * p0
        + 16.0 * p1 * chi_tilde_parallel * (chi_tilde_parallel + 1.0)
        + eisco
        - 1.0
    )
    return float(1.0 - radiated)


def paper_nr_final_spin_magnitude(
    q: float, spin1: Any, spin2: Any, orbital_axis: Any
) -> float:
    """Hofmann--Barausse--Rezzolla 2016 HBR16_34corr fit."""
    np, _ = _require_numpy_h5py()
    axis = _unit_vector(orbital_axis, "orbital angular-momentum direction")
    spin1 = np.asarray(spin1, dtype=np.float64)
    spin2 = np.asarray(spin2, dtype=np.float64)
    chi1 = float(np.linalg.norm(spin1))
    chi2 = float(np.linalg.norm(spin2))
    theta1, theta2, _ = _spin_angles(spin1, spin2, axis)
    if chi1 > 0.0 and chi2 > 0.0:
        cos_theta12 = float(np.clip(np.dot(spin1, spin2) / (chi1 * chi2), -1.0, 1.0))
    else:
        cos_theta12 = 1.0

    # HBR16 Table I, n_M=3, n_J=4.  K00 is fixed by their Eq. (11).
    kfit = np.asarray(
        [
            [0.0, 3.39221, 4.48865, -5.77101, -13.0459],
            [35.1278, -72.9336, -86.0036, 93.7371, 200.975],
            [-146.822, 387.184, 447.009, -467.383, -884.339],
            [223.911, -648.502, -697.177, 753.738, 1166.89],
        ],
        dtype=np.float64,
    )
    kfit[0, 0] = 16.0 * (
        0.68646
        - sum(kfit[row, 0] / 4.0 ** (row + 2) for row in range(1, 4))
        - math.sqrt(3.0) / 2.0
    )
    eta = q / (1.0 + q) ** 2
    theta1_corrected = theta1 + 0.024 * math.sin(theta1)
    theta2_corrected = theta2 + 0.024 * math.sin(theta2)
    aligned1 = chi1 * math.cos(theta1_corrected)
    aligned2 = chi2 * math.cos(theta2_corrected)
    atot = (aligned1 + q**2 * aligned2) / (1.0 + q) ** 2
    aeff = atot + 0.474046 * eta * (aligned1 + aligned2)
    if abs(aeff) > 1.0 + 1.0e-12:
        raise ValueError("HBR16 effective spin lies outside the Kerr interval")
    aeff = float(np.clip(aeff, -1.0, 1.0))
    z1 = 1.0 + np.cbrt(1.0 - aeff**2) * (
        np.cbrt(1.0 + aeff) + np.cbrt(1.0 - aeff)
    )
    z2 = math.sqrt(3.0 * aeff**2 + z1**2)
    spin_sign = 0.0 if aeff == 0.0 else math.copysign(1.0, aeff)
    risco = 3.0 + z2 - spin_sign * math.sqrt(
        max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    )
    eisco = math.sqrt(1.0 - 2.0 / (3.0 * risco))
    lisco = 2.0 / (3.0 * math.sqrt(3.0)) * (
        1.0 + 2.0 * math.sqrt(3.0 * risco - 2.0)
    )
    polynomial = 0.0
    for i in range(kfit.shape[0]):
        for j in range(kfit.shape[1]):
            polynomial += kfit[i, j] * eta ** (i + 1) * aeff**j
    ell = abs(lisco - 2.0 * atot * (eisco - 1.0) + polynomial)
    radicand = (
        chi1**2
        + chi2**2 * q**4
        + 2.0 * chi1 * chi2 * q**2 * cos_theta12
        + 2.0 * (aligned1 + q**2 * aligned2) * ell * q
        + (ell * q) ** 2
    )
    return float(min(math.sqrt(max(0.0, radicand)) / (1.0 + q) ** 2, 1.0))


def paper_nr_kick_components(
    q: float,
    spin1: Any,
    spin2: Any,
    orbital_axis: Any,
    merger_phase: float,
) -> Any:
    """Gerosa--Kesden 2016 collected NR kick fit in an L-aligned frame."""
    np, _ = _require_numpy_h5py()
    axis = _unit_vector(orbital_axis, "orbital angular-momentum direction")
    spin1 = np.asarray(spin1, dtype=np.float64)
    spin2 = np.asarray(spin2, dtype=np.float64)
    eta = q / (1.0 + q) ** 2
    delta = (spin1 - q * spin2) / (1.0 + q)
    delta_parallel = float(np.dot(delta, axis))
    delta_perpendicular = float(np.linalg.norm(np.cross(delta, axis)))
    chi_tilde = (spin1 + q**2 * spin2) / (1.0 + q) ** 2
    chi_tilde_parallel = float(np.dot(chi_tilde, axis))
    chi_tilde_perpendicular = float(np.linalg.norm(np.cross(chi_tilde, axis)))

    zeta = math.radians(145.0)
    v_mass = 1.2e4 * eta**2 * (1.0 - 0.93 * eta) * (1.0 - q) / (1.0 + q)
    v_perpendicular = 6.9e3 * eta**2 * delta_parallel
    v_parallel = 16.0 * eta**2 * (
        delta_perpendicular
        * (
            3677.76
            + 2.0 * 2481.21 * chi_tilde_parallel
            + 4.0 * 1792.45 * chi_tilde_parallel**2
            + 8.0 * 1506.52 * chi_tilde_parallel**3
        )
        + chi_tilde_perpendicular
        * delta_parallel
        * (2.0 * 1140.0 + 4.0 * 2481.0 * chi_tilde_parallel)
    ) * math.cos(merger_phase)
    return np.asarray(
        [
            v_mass + v_perpendicular * math.cos(zeta),
            v_perpendicular * math.sin(zeta),
            v_parallel,
        ],
        dtype=np.float64,
    ) / SPEED_OF_LIGHT_KM_S


def resolve_paper_nr_fit(
    args: argparse.Namespace,
    raw: dict[str, Any],
    *,
    default_fit_time_M: float | None = None,
    default_time_policy: str = "transition-start-paper-prescription",
) -> dict[str, Any]:
    """Evaluate the paper's three remnant fits on one recorded PN sample."""
    np, _ = _require_numpy_h5py()
    if "orbital_angular_momentum" not in raw:
        raise ValueError(
            "--remnant-model=paper-nr-fit requires CBwaves Lx,Ly,Lz columns"
        )
    raw_time = np.asarray(raw["time"], dtype=np.float64)
    merger_time = float(
        raw_time[-1] if args.merger_time_M is None else args.merger_time_M
    )
    if args.remnant_fit_time_M is None:
        if default_fit_time_M is None:
            default_fit_time_M = resolve_transition(args, raw_time)[0]
        requested_time = float(default_fit_time_M)
        requested_time_policy = default_time_policy
    else:
        requested_time = float(args.remnant_fit_time_M)
        requested_time_policy = "explicit-cli"
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(merger_time))
    if requested_time > merger_time + tolerance:
        raise ValueError("--remnant-fit-time-M cannot lie after merger time")
    sample_index = int(np.searchsorted(raw_time, requested_time, side="right") - 1)
    if sample_index < 0 or sample_index >= len(raw_time):
        raise ValueError("remnant fit time lies outside the CBwaves table")

    spin = np.asarray(raw["spin_chi"][sample_index], dtype=np.float64)
    orbital_l = np.asarray(raw["orbital_angular_momentum"][sample_index], dtype=np.float64)
    orbital_axis = _unit_vector(orbital_l, "CBwaves orbital angular momentum")
    relative_position = np.asarray(
        raw["position"][sample_index, 0] - raw["position"][sample_index, 1],
        dtype=np.float64,
    )
    in_plane_position = relative_position - np.dot(relative_position, orbital_axis) * orbital_axis
    if np.linalg.norm(in_plane_position) <= 1.0e-12:
        relative_velocity = np.asarray(
            raw["velocity"][sample_index, 0] - raw["velocity"][sample_index, 1],
            dtype=np.float64,
        )
        in_plane_position = relative_velocity - np.dot(relative_velocity, orbital_axis) * orbital_axis
    radial_axis = _unit_vector(in_plane_position, "instantaneous orbital radial axis")
    tangential_axis = _unit_vector(
        np.cross(orbital_axis, radial_axis), "instantaneous orbital tangential axis"
    )
    # Re-orthogonalize radial_axis so the stored basis is right-handed to roundoff.
    radial_axis = _unit_vector(np.cross(tangential_axis, orbital_axis), "orbital radial axis")
    orbital_basis = np.stack((radial_axis, tangential_axis, orbital_axis))

    q = args.mass2_fraction / args.mass1_fraction
    mass = paper_nr_final_mass(q, spin[0], spin[1], orbital_axis)
    spin_magnitude = paper_nr_final_spin_magnitude(
        q, spin[0], spin[1], orbital_axis
    )
    # HBR16 assumes the final-spin direction follows total J at plunge.  The
    # archived wrapper uses r=3M, so preserve that radius but compute the full
    # 3-vector instead of rotating around a privileged global z axis.
    eta = q / (1.0 + q) ** 2
    total_j = (
        eta * math.sqrt(3.0) * orbital_axis
        + args.mass1_fraction**2 * spin[0]
        + args.mass2_fraction**2 * spin[1]
    )
    final_spin_axis = _unit_vector(total_j, "plunge total angular momentum")
    final_spin = spin_magnitude * final_spin_axis
    kick_orbital = paper_nr_kick_components(
        q, spin[0], spin[1], orbital_axis, args.remnant_kick_phase_rad
    )
    kick = kick_orbital @ orbital_basis
    if not math.isfinite(mass) or not 0.0 < mass <= 1.0:
        raise ValueError("paper NR final-mass fit returned a value outside (0,1]")
    if not math.isfinite(spin_magnitude) or not 0.0 <= spin_magnitude < 1.0:
        raise ValueError(
            "paper NR final-spin fit returned an extremal/super-extremal value"
        )
    if not np.all(np.isfinite(kick)) or np.linalg.norm(kick) >= 1.0:
        raise ValueError("paper NR kick fit returned a nonfinite/luminal result")
    theta1, theta2, delta_phi = _spin_angles(spin[0], spin[1], orbital_axis)
    return {
        "mass": mass,
        "spin_chi": final_spin,
        "spin_a": mass * final_spin,
        "kick": kick,
        "model": "paper-nr-fit",
        "fit": {
            "version": REMNANT_FIT_VERSION,
            "mass_fit": "Barausse-Morozova-Rezzolla-2012-Eq18",
            "spin_fit": "Hofmann-Barausse-Rezzolla-2016-HBR16_34corr",
            "spin_direction": "unit-J-at-plunge-r=3M",
            "kick_fit": "Gerosa-Kesden-2016-collected-NR-fits",
            "kick_terms": (
                "mass-asymmetry,aligned-spin,superkick,hangup-kick,cross-kick;"
                "all enabled"
            ),
            "kick_phase_policy": "explicit-radians-no-random-sampling",
            "kick_zero_spin_policy": "analytic continuous zero-spin limit",
            "reference_precession_sha256": PRECESSION_REFERENCE_SHA256,
            "sample_index": sample_index,
            "sample_time": float(raw_time[sample_index]),
            "requested_time": requested_time,
            "requested_time_policy": requested_time_policy,
            "sample_separation": float(raw["separation"][sample_index]),
            "mass_ratio_q_m2_over_m1": q,
            "theta1": theta1,
            "theta2": theta2,
            "delta_phi": delta_phi,
            "kick_phase_rad": float(args.remnant_kick_phase_rad),
            "mass_fraction": np.asarray(
                [args.mass1_fraction, args.mass2_fraction], dtype=np.float64
            ),
            "spin_chi": spin,
            "orbital_angular_momentum": orbital_l,
            "orbital_basis": orbital_basis,
            "kick_components_orbital": kick_orbital,
        },
    }


def resolve_remnant(
    args: argparse.Namespace,
    raw: dict[str, Any] | None = None,
    *,
    default_orientation_time_M: float | None = None,
) -> dict[str, Any]:
    np, _ = _require_numpy_h5py()
    if args.remnant_model == "equal-nonspinning":
        if (
            abs(args.mass1_fraction - 0.5) > 1.0e-14
            or abs(args.mass2_fraction - 0.5) > 1.0e-14
            or np.linalg.norm(args.spin1) > 1.0e-14
            or np.linalg.norm(args.spin2) > 1.0e-14
        ):
            raise ValueError(
                "the equal-nonspinning remnant preset requires m1=m2=0.5 and zero "
                "initial spins; use --remnant-model=paper-nr-fit or "
                "--remnant-model=explicit otherwise"
            )
        mass = 0.95173
        orbital_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        orientation: dict[str, Any] | None = None
        if raw is not None:
            reference_time = (
                resolve_transition(args, raw["time"])[0]
                if default_orientation_time_M is None
                else float(default_orientation_time_M)
            )
            sample_index = int(
                np.searchsorted(raw["time"], reference_time, side="right") - 1
            )
            if sample_index < 0 or sample_index >= len(raw["time"]):
                raise ValueError(
                    "equal-nonspinning remnant orientation time lies outside "
                    "the CBwaves table"
                )
            if "orbital_angular_momentum" in raw:
                orbital_l = np.asarray(
                    raw["orbital_angular_momentum"][sample_index],
                    dtype=np.float64,
                )
                source = "recorded CBwaves orbital angular momentum"
            else:
                relative_position = (
                    raw["position"][sample_index, 0]
                    - raw["position"][sample_index, 1]
                )
                relative_velocity = (
                    raw["velocity"][sample_index, 0]
                    - raw["velocity"][sample_index, 1]
                )
                orbital_l = np.cross(relative_position, relative_velocity)
                source = "specific relative orbital angular momentum r_cross_v"
            orbital_axis = _unit_vector(
                orbital_l, "equal-nonspinning remnant orbital axis"
            )
            orientation = {
                "policy": "aligned-with-orbital-L-at-transition-start",
                "source": source,
                "requested_time": float(reference_time),
                "sample_index": sample_index,
                "sample_time": float(raw["time"][sample_index]),
                "orbital_angular_momentum": orbital_l,
                "orbital_axis": orbital_axis,
            }
        spin_chi = 0.68646 * orbital_axis
        kick = np.zeros(3, dtype=np.float64)
        model = "deterministic equal-mass nonspinning orbital-L-aligned reference"
    elif args.remnant_model == "explicit":
        if args.remnant_mass is None or args.remnant_spin_chi is None:
            raise ValueError(
                "--remnant-model=explicit requires --remnant-mass and "
                "--remnant-spin-chi X Y Z"
            )
        mass = float(args.remnant_mass)
        spin_chi = np.asarray(args.remnant_spin_chi, dtype=np.float64)
        kick = np.asarray(
            (0.0, 0.0, 0.0) if args.remnant_kick is None else args.remnant_kick,
            dtype=np.float64,
        )
        model = "user-supplied deterministic remnant"
    else:
        if raw is None:
            raise ValueError("paper-nr-fit requires a loaded CBwaves trajectory")
        return resolve_paper_nr_fit(args, raw)
    if not 0.0 < mass <= 1.0:
        raise ValueError("remnant mass must lie in (0,1] M_ref")
    if not np.all(np.isfinite(spin_chi)) or np.linalg.norm(spin_chi) >= 1.0:
        raise ValueError("remnant dimensionless spin must be finite with norm < 1")
    if not np.all(np.isfinite(kick)) or np.linalg.norm(kick) >= 1.0:
        raise ValueError("remnant kick must be finite and subluminal")
    result = {
        "mass": mass,
        "spin_chi": spin_chi,
        "spin_a": mass * spin_chi,
        "kick": kick,
        "model": model,
    }
    if args.remnant_model == "equal-nonspinning" and orientation is not None:
        result["orientation"] = orientation
    return result


def validate_raw_remnant_compatibility(
    args: argparse.Namespace, raw: dict[str, Any]
) -> None:
    """Reject a remnant preset whose assumptions disagree with the DAT."""
    np, _ = _require_numpy_h5py()
    if args.remnant_model != "equal-nonspinning":
        return
    maximum_spin = float(np.max(np.linalg.norm(raw["spin_chi"], axis=2)))
    if maximum_spin > 1.0e-10:
        raise ValueError(
            "the equal-nonspinning remnant preset is incompatible with the "
            f"actual DAT spin columns (max |chi|={maximum_spin:.17g}); use "
            "--remnant-model=explicit with a justified remnant fit"
        )


def resolve_transition(
    args: argparse.Namespace, raw_time: Any
) -> tuple[float, float, float]:
    np, _ = _require_numpy_h5py()
    merger = float(raw_time[-1] if args.merger_time_M is None else args.merger_time_M)
    have_start = args.transition_start_M is not None
    have_end = args.transition_end_M is not None
    if have_start != have_end:
        raise ValueError("set both --transition-start-M and --transition-end-M")
    if have_start:
        start = float(args.transition_start_M)
        end = float(args.transition_end_M)
    else:
        start = merger - args.transition_half_width_M
        end = merger + args.transition_half_width_M
    if not float(raw_time[0]) <= start < merger < end:
        raise ValueError(
            "transition must satisfy first_source_time <= start < merger_time < end"
        )
    if merger > float(raw_time[-1]) + 32.0 * np.finfo(float).eps * max(1.0, abs(merger)):
        raise ValueError("merger time cannot lie after the final CBwaves sample")
    return start, merger, end


def merger_reach_diagnostic(
    raw: dict[str, Any], args: argparse.Namespace
) -> dict[str, float | bool | str]:
    """Prove that the PN state actually used at merger reached ``rmin``.

    The acceptance tolerance is a fixed numerical roundoff allowance.  It is
    deliberately independent of ``print_step`` and velocity, so a sparse DAT
    cadence cannot turn a tmax-limited inspiral into a completed trajectory.
    """
    np, _ = _require_numpy_h5py()
    raw_time = np.asarray(raw["time"], dtype=np.float64)
    merger_time = float(
        raw_time[-1] if args.merger_time_M is None else args.merger_time_M
    )
    time_tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, abs(merger_time), abs(float(raw_time[-1]))
    )
    if merger_time < float(raw_time[0]) - time_tolerance or merger_time > float(
        raw_time[-1]
    ) + time_tolerance:
        raise ValueError("merger time lies outside the CBwaves table")
    query = np.asarray([merger_time], dtype=np.float64)
    if args.transition_velocity == "consistent":
        selected_position, _ = interpolate_cubic_hermite(
            raw_time, raw["position"], raw["velocity"], query
        )
        interpolation = "cubic-Hermite position"
    else:
        selected_position = interpolate_components(
            raw_time, raw["position"], query
        )
        interpolation = "linear position"
    selected_separation = float(
        np.linalg.norm(selected_position[0, 0] - selected_position[0, 1])
    )
    final_separation = float(raw["separation"][-1])
    output_cadence = args.print_step * args.dt_M
    tolerance = max(
        MERGER_REACH_ABSOLUTE_TOLERANCE_M,
        256.0
        * np.finfo(float).eps
        * max(1.0, abs(args.rmin_M), abs(selected_separation)),
    )
    threshold = args.rmin_M + tolerance
    return {
        "reached": selected_separation <= threshold,
        "final_separation": final_separation,
        "selected_merger_separation": selected_separation,
        "evaluation_time": merger_time,
        "requested_separation": float(args.rmin_M),
        "numerical_tolerance": tolerance,
        "acceptance_threshold": threshold,
        "output_cadence": output_cadence,
        "interpolation": interpolation,
    }


def validate_merger_reached(
    raw: dict[str, Any], args: argparse.Namespace
) -> dict[str, float | bool | str]:
    """Reject a tmax-limited run that would otherwise be mislabeled merger."""
    diagnostic = merger_reach_diagnostic(raw, args)
    if (
        not diagnostic["reached"] and not args.allow_truncated_inspiral
    ):
        raise ValueError(
            "the selected merger state did not reach the requested merger "
            f"separation: r12(t_merger)="
            f"{diagnostic['selected_merger_separation']:.8g} M at "
            f"t={diagnostic['evaluation_time']:.8g} M, "
            f"rmin={args.rmin_M:.8g} M, fixed numerical "
            f"tolerance={diagnostic['numerical_tolerance']:.3g} M. Increase "
            "--tmax-M or use "
            "--allow-truncated-inspiral only for diagnostic tests."
        )
    return diagnostic


def extend_to_exact_remnant(
    raw: dict[str, Any],
    args: argparse.Namespace,
    remnant: dict[str, Any],
    *,
    minimum_auto_half_width_M: float = 0.0,
) -> dict[str, Any]:
    np, _ = _require_numpy_h5py()
    raw_time = raw["time"]
    requested_start, merger, requested_end = resolve_transition(args, raw_time)
    output_dt = args.output_dt_M
    if output_dt is None:
        output_dt = float(np.median(np.diff(raw_time)))
    if not math.isfinite(output_dt) or output_dt <= 0.0:
        raise ValueError("output timestep must be positive and finite")
    explicit_window = args.transition_start_M is not None

    def construct(start: float, end: float) -> dict[str, Any]:
        source_times = np.unique(
            np.concatenate((raw_time[raw_time < merger], np.asarray([merger])))
        )
        final_time = end + args.post_merger_duration_M
        appended = np.arange(
            merger + output_dt, final_time, output_dt, dtype=np.float64
        )
        time = np.unique(
            np.concatenate(
                (
                    source_times,
                    appended,
                    np.asarray([start, merger, end, final_time], dtype=np.float64),
                )
            )
        )
        time = time[(time >= raw_time[0]) & (time <= final_time)]
        if len(time) < 4 or not np.all(np.diff(time) > 0.0):
            raise ValueError("constructed output time grid is invalid")

        query = np.minimum(time, merger)
        if args.transition_velocity == "consistent":
            position0, velocity0 = interpolate_cubic_hermite(
                raw_time, raw["position"], raw["velocity"], query
            )
        else:
            position0 = interpolate_components(raw_time, raw["position"], query)
            velocity0 = interpolate_components(raw_time, raw["velocity"], query)
        spin_chi0 = interpolate_components(raw_time, raw["spin_chi"], query)
        if args.transition_velocity == "consistent":
            merger_position, merger_velocity = interpolate_cubic_hermite(
                raw_time,
                raw["position"],
                raw["velocity"],
                np.asarray([merger]),
            )
        else:
            merger_position = interpolate_components(
                raw_time, raw["position"], np.asarray([merger])
            )
            merger_velocity = interpolate_components(
                raw_time, raw["velocity"], np.asarray([merger])
            )
        merger_spin = interpolate_components(
            raw_time, raw["spin_chi"], np.asarray([merger])
        )[0]
        tail = time > merger
        if np.any(tail):
            position0[tail] = merger_position[0] + (
                (time[tail] - merger)[:, None, None]
                * merger_velocity[0][None, :, :]
            )
            velocity0[tail] = merger_velocity[0]
            spin_chi0[tail] = merger_spin

        window = smooth_window(time, start, end)
        window_derivative = smooth_window_derivative(time, start, end)
        one_minus = 1.0 - window
        initial_mass = np.asarray(
            [args.mass1_fraction, args.mass2_fraction], dtype=np.float64
        )
        mass = one_minus[:, None] * initial_mass[None, :] + (
            0.5 * remnant["mass"] * window[:, None]
        )
        spin_a0 = spin_chi0 * initial_mass[None, :, None]
        spin_a = one_minus[:, None, None] * spin_a0 + (
            window[:, None, None] * remnant["spin_a"][None, None, :]
        )
        spin_chi = spin_a / mass[:, :, None]

        if args.transition_velocity == "consistent":
            position_at_start, _ = interpolate_cubic_hermite(
                raw_time,
                raw["position"],
                raw["velocity"],
                np.asarray([start]),
            )
        else:
            position_at_start = interpolate_components(
                raw_time, raw["position"], np.asarray([start])
            )
        anchor = np.sum(initial_mass[:, None] * position_at_start[0], axis=0)
        remnant_worldline = anchor[None, :] + (
            (time - start)[:, None] * remnant["kick"][None, :]
        )
        position = one_minus[:, None, None] * position0 + (
            window[:, None, None] * remnant_worldline[:, None, :]
        )
        paper_blended_velocity = one_minus[:, None, None] * velocity0 + (
            window[:, None, None] * remnant["kick"][None, None, :]
        )
        consistent_velocity = paper_blended_velocity + (
            window_derivative[:, None, None]
            * (remnant_worldline[:, None, :] - position0)
        )
        velocity = (
            consistent_velocity
            if args.transition_velocity == "consistent"
            else paper_blended_velocity
        )
        speed = np.linalg.norm(velocity, axis=2)
        discrete_derivative = np.gradient(position, time, axis=0, edge_order=2)
        discrete_velocity_error = np.linalg.norm(
            discrete_derivative - consistent_velocity, axis=2
        )
        arrays = (
            time,
            position,
            velocity,
            mass,
            spin_chi,
            spin_a,
            window,
            window_derivative,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("nonfinite value produced during merger transition")
        exact = time >= end
        if not np.any(exact):
            raise ValueError("output contains no exact-remnant samples")
        tolerance = 64.0 * np.finfo(float).eps
        if (
            np.max(np.abs(position[exact, 0] - position[exact, 1])) > tolerance
            or np.max(np.abs(velocity[exact, 0] - velocity[exact, 1])) > tolerance
            or np.max(np.abs(spin_a[exact, 0] - spin_a[exact, 1])) > tolerance
            or np.max(np.abs(mass[exact] - 0.5 * remnant["mass"])) > tolerance
        ):
            raise AssertionError(
                "post-transition superposed terms did not collapse exactly"
            )
        return {
            "time": time,
            "position": position,
            "velocity": velocity,
            "paper_blended_velocity": paper_blended_velocity,
            "consistent_velocity": consistent_velocity,
            "discrete_velocity_error": discrete_velocity_error,
            "maximum_speed": float(np.max(speed)),
            "mass": mass,
            "spin_chi": spin_chi,
            "spin_a": spin_a,
            "window": window,
            "window_derivative": window_derivative,
            "separation": np.linalg.norm(position[:, 0] - position[:, 1], axis=1),
            "transition_start": start,
            "merger_time": merger,
            "transition_end": end,
            "remnant_worldline_anchor": anchor,
            "remnant_worldline": remnant_worldline,
        }

    start = requested_start
    end = requested_end
    auto_expansions = 0
    if not explicit_window and minimum_auto_half_width_M > 0.0:
        available_half_width = merger - float(raw_time[0])
        while max(merger - start, end - merger) < minimum_auto_half_width_M:
            current_half_width = max(merger - start, end - merger)
            next_half_width = min(2.0 * current_half_width, available_half_width)
            if next_half_width <= current_half_width * (
                1.0 + 8.0 * np.finfo(float).eps
            ):
                raise ValueError(
                    "cannot preserve the transition width required to evaluate "
                    "the remnant at the actual transition start"
                )
            start = merger - next_half_width
            end = merger + next_half_width
            auto_expansions += 1
    while True:
        result = construct(start, end)
        maximum_speed = result["maximum_speed"]
        speed_ok = maximum_speed <= args.transition_max_speed
        if args.transition_velocity != "consistent" or speed_ok:
            break
        if explicit_window or not args.auto_expand_transition:
            raise ValueError(
                "worldline-consistent merger transition reaches "
                f"|v|={maximum_speed:.6g}, above the configured "
                f"limit {args.transition_max_speed:.6g}; widen the transition "
                "or explicitly use --transition-velocity=paper-blend only for "
                "paper-reproduction work"
            )
        current_half_width = max(merger - start, end - merger)
        available_half_width = merger - float(raw_time[0])
        next_half_width = min(2.0 * current_half_width, available_half_width)
        if next_half_width <= current_half_width * (1.0 + 8.0 * np.finfo(float).eps):
            raise ValueError(
                "cannot auto-expand the worldline-consistent transition enough "
                f"to keep |v| <= {args.transition_max_speed:.6g}; provide an "
                "earlier CBwaves trajectory or use an explicit wider interval"
            )
        start = merger - next_half_width
        end = merger + next_half_width
        auto_expansions += 1

    maximum_speed = result["maximum_speed"]
    if maximum_speed >= 1.0:
        raise ValueError(
            f"selected boost velocity is luminal/superluminal: |v|max="
            f"{maximum_speed:.6g}"
        )
    result["requested_transition_start"] = requested_start
    result["requested_transition_end"] = requested_end
    result["transition_auto_expansions"] = auto_expansions
    result["worldline_velocity_consistent"] = (
        args.transition_velocity == "consistent"
    )
    return result


def resolve_remnant_and_trajectory(
    args: argparse.Namespace, raw: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve mutually consistent remnant inputs and transition geometry."""
    np, _ = _require_numpy_h5py()
    if args.remnant_model != "paper-nr-fit":
        remnant = resolve_remnant(args, raw)
        trajectory = extend_to_exact_remnant(raw, args, remnant)
        if args.remnant_model == "equal-nonspinning":
            actual_start = float(trajectory["transition_start"])
            orientation_time = float(remnant["orientation"]["requested_time"])
            tolerance = 64.0 * np.finfo(float).eps * max(
                1.0, abs(actual_start), abs(orientation_time)
            )
            if abs(actual_start - orientation_time) > tolerance:
                remnant = resolve_remnant(
                    args,
                    raw,
                    default_orientation_time_M=actual_start,
                )
                trajectory = extend_to_exact_remnant(
                    raw,
                    args,
                    remnant,
                    minimum_auto_half_width_M=(
                        float(trajectory["merger_time"]) - actual_start
                    ),
                )
        return remnant, trajectory

    if args.remnant_fit_time_M is not None:
        remnant = resolve_paper_nr_fit(args, raw)
        return remnant, extend_to_exact_remnant(raw, args, remnant)

    requested_start, merger, _ = resolve_transition(args, raw["time"])
    fit_time = requested_start
    minimum_half_width = 0.0
    for _ in range(32):
        remnant = resolve_paper_nr_fit(
            args,
            raw,
            default_fit_time_M=fit_time,
            default_time_policy="actual-transition-start-paper-prescription",
        )
        trajectory = extend_to_exact_remnant(
            raw,
            args,
            remnant,
            minimum_auto_half_width_M=minimum_half_width,
        )
        actual_start = float(trajectory["transition_start"])
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(actual_start), abs(fit_time)
        )
        if abs(actual_start - fit_time) <= tolerance:
            return remnant, trajectory
        minimum_half_width = max(minimum_half_width, merger - actual_start)
        fit_time = actual_start
    raise ValueError(
        "remnant fit and automatically expanded transition start did not converge"
    )


def dataset(group: Any, name: str, data: Any, units: str, description: str) -> Any:
    created = group.create_dataset(name, data=data, compression="gzip", shuffle=True)
    created.attrs["units"] = units
    created.attrs["description"] = description
    return created


def write_native_hdf5(
    output: Path,
    *,
    raw: dict[str, Any],
    trajectory: dict[str, Any],
    remnant: dict[str, Any],
    args: argparse.Namespace,
    provenance: dict[str, str],
    ini_text: str,
) -> None:
    np, h5py = _require_numpy_h5py()
    generator_sha256 = sha256_file(Path(__file__).resolve())
    source_is_verified = args.convert_existing_dat is None
    trajectory_model = (
        "paper_cbwaves_4pn_local"
        if source_is_verified
        else "external_cbwaves_dat_unverified"
    )
    pn_terms_metadata = (
        ",".join(PN_TERMS) if source_is_verified else "unknown-external-dat"
    )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        interpolation = (
            "cubic_hermite_position_velocity"
            if trajectory["worldline_velocity_consistent"]
            else "linear"
        )
        boost_velocity_model = (
            "derivative_of_position"
            if trajectory["worldline_velocity_consistent"]
            else "paper_eq16_independent_blend"
        )
        with h5py.File(temporary, "w") as handle:
            trajectory_group = handle.create_group("trajectory")
            trajectory_group.attrs["schema"] = SCHEMA_NAME
            trajectory_group.attrs["schema_version"] = np.int32(SCHEMA_VERSION)
            trajectory_group.attrs["units"] = "G=c=M_ref=1"
            trajectory_group.attrs["position_gauge"] = "harmonic"
            trajectory_group.attrs["spin_convention"] = "kerr_a"
            trajectory_group.attrs["interpolation"] = interpolation
            trajectory_group.attrs["generator"] = (
                "scripts/generate_paper_sks_trajectory.py"
            )
            trajectory_group.attrs["generator_version"] = GENERATOR_VERSION
            trajectory_group.attrs["merger_reach_contract"] = (
                MERGER_REACH_CONTRACT
            )
            trajectory_group.attrs["trajectory_model"] = trajectory_model
            trajectory_group.attrs["pn_terms"] = pn_terms_metadata
            trajectory_group.attrs["merger_separation_reached"] = np.uint8(
                raw["merger_reach"]["reached"]
            )
            trajectory_group.attrs["trajectory_status"] = raw["trajectory_status"]
            trajectory_group.attrs["worldline_velocity_consistent"] = np.uint8(
                trajectory["worldline_velocity_consistent"]
            )
            trajectory_group.attrs["boost_velocity_model"] = boost_velocity_model
            handle.attrs["schema_name"] = SCHEMA_NAME
            handle.attrs["schema_version"] = np.uint32(SCHEMA_VERSION)
            handle.attrs["units"] = "G=c=M_ref=1"
            handle.attrs["M_ref"] = np.float64(1.0)
            handle.attrs["M_ref_solar_mass"] = np.float64(
                args.dat_total_mass_solar
                if args.convert_existing_dat is not None
                else args.reference_mass_solar
            )
            handle.attrs["coordinate_system"] = "global_cartesian"
            handle.attrs["trajectory_gauge"] = "harmonic"
            handle.attrs["metric_coordinates"] = "cartesian_kerr_schild"
            handle.attrs["time_origin"] = "CBwaves integration start"
            handle.attrs["component_order"] = "axis 1 is [BH1 (heavier/equal), BH2]"
            handle.attrs["mass_ratio_definition"] = "q=m2/m1 in (0,1]"
            handle.attrs["mass_ratio"] = np.float64(
                args.mass2_fraction / args.mass1_fraction
            )
            handle.attrs["out_of_range_policy"] = (
                "error below t_min; above t_max permit only analytic uniform-boost "
                "extension of a validated exact-W=1 native remnant tail"
            )
            handle.attrs["future_extension_policy"] = (
                "validated native exact-remnant tail: constant masses/spins/velocity "
                "and linear common worldline; otherwise error"
            )
            handle.attrs["interpolation"] = interpolation
            handle.attrs["worldline_velocity_consistent"] = np.uint8(
                trajectory["worldline_velocity_consistent"]
            )
            handle.attrs["boost_velocity_model"] = boost_velocity_model
            handle.attrs["velocity_semantics"] = (
                "analytic d(position)/dt including dW/dt; runtime uses coupled cubic Hermite position/velocity interpolation"
                if trajectory["worldline_velocity_consistent"]
                else "paper v3 Eq. (16) algebraic boost; not d(position)/dt during transition"
            )
            handle.attrs["generator"] = "scripts/generate_paper_sks_trajectory.py"
            handle.attrs["generator_version"] = GENERATOR_VERSION
            handle.attrs["merger_reach_contract"] = MERGER_REACH_CONTRACT
            handle.attrs["generator_sha256"] = generator_sha256
            handle.attrs["trajectory_model"] = trajectory_model
            handle.attrs["remnant_model"] = args.remnant_model
            handle.attrs["merger_separation_reached"] = np.uint8(
                raw["merger_reach"]["reached"]
            )
            handle.attrs["trajectory_status"] = raw["trajectory_status"]
            handle.attrs["final_pn_separation"] = np.float64(
                raw["merger_reach"]["final_separation"]
            )
            handle.attrs["selected_merger_separation"] = np.float64(
                raw["merger_reach"]["selected_merger_separation"]
            )
            handle.attrs["merger_reach_evaluation_time"] = np.float64(
                raw["merger_reach"]["evaluation_time"]
            )
            handle.attrs["merger_reach_interpolation"] = raw["merger_reach"][
                "interpolation"
            ]
            handle.attrs["requested_merger_separation"] = np.float64(
                raw["merger_reach"]["requested_separation"]
            )
            handle.attrs["merger_separation_numerical_tolerance"] = np.float64(
                raw["merger_reach"]["numerical_tolerance"]
            )
            handle.attrs["merger_separation_acceptance_threshold"] = np.float64(
                raw["merger_reach"]["acceptance_threshold"]
            )
            handle.attrs["source_verified"] = np.uint8(source_is_verified)
            if source_is_verified:
                handle.attrs["source_doi"] = ZENODO_DOI
                handle.attrs["source_license_record"] = "CC-BY-4.0"
                handle.attrs["source_license_scope"] = (
                    "license stated on Zenodo record; bundled third-party files may "
                    "carry separate notices and are not redistributed in this HDF5"
                )
            else:
                handle.attrs["source_provenance"] = (
                    "unverified external DAT supplied by caller; no source DOI, license, "
                    "revision, or paper-model identity is asserted"
                )
                handle.attrs["external_dat_semantics_asserted_by_user"] = np.uint8(
                    args.assume_cbwaves_harmonic
                )
            handle.attrs["pn_term_set"] = pn_terms_metadata
            handle.attrs["pn_4pn_scope"] = (
                "patched CBwaves local instantaneous polynomial; excludes harmonic "
                "log terms, conservative nonlocal tail, and 4PN dissipative tail"
                if source_is_verified
                else "unknown; caller-supplied external DAT was not source-verified"
            )
            handle.attrs["window_formula"] = (
                "W=E(u)/(E(u)+E(1-u)), E(x)=exp(-1/x) for x>0 else 0"
            )
            handle.attrs["requested_transition_start"] = np.float64(
                trajectory["requested_transition_start"]
            )
            handle.attrs["requested_transition_end"] = np.float64(
                trajectory["requested_transition_end"]
            )
            handle.attrs["transition_auto_expansions"] = np.uint32(
                trajectory["transition_auto_expansions"]
            )
            handle.attrs["maximum_boost_speed"] = np.float64(
                trajectory["maximum_speed"]
            )

            dataset(
                trajectory_group,
                "t",
                trajectory["time"],
                "M_ref",
                "trajectory time",
            )
            dataset(
                trajectory_group,
                "position",
                trajectory["position"],
                "M_ref",
                "coordinate centers, shape [time, hole, xyz]",
            )
            dataset(
                trajectory_group,
                "velocity",
                trajectory["velocity"],
                "c",
                "boost velocities, shape [time, hole, xyz]",
            )
            dataset(
                trajectory_group,
                "mass",
                trajectory["mass"],
                "M_ref",
                "SKS term masses",
            )
            dataset(
                trajectory_group,
                "spin_chi",
                trajectory["spin_chi"],
                "dimensionless",
                "per-term a/M; may exceed one during exact-remnant representation",
            )
            dataset(
                trajectory_group,
                "kerr_a",
                trajectory["spin_a"],
                "M_ref",
                "dimensional Kerr vectors used by the SKS metric",
            )
            dataset(
                trajectory_group,
                "merger_weight",
                trajectory["window"],
                "dimensionless",
                "merger window W",
            )
            dataset(
                trajectory_group,
                "separation",
                trajectory["separation"],
                "M_ref",
                "distance between the two coordinate centers",
            )

            # Stable, zero-copy hard-link aliases retain the concise canonical
            # names used in the physics documentation while /trajectory is the
            # exact contract consumed by the C++ reader.
            for alias, target in (
                ("time", "t"),
                ("position", "position"),
                ("velocity", "velocity"),
                ("mass", "mass"),
                ("spin_chi", "spin_chi"),
                ("spin_a", "kerr_a"),
                ("window", "merger_weight"),
                ("separation", "separation"),
            ):
                handle[alias] = trajectory_group[target]

            diagnostics = handle.create_group("diagnostics")
            dataset(
                diagnostics,
                "paper_blended_velocity",
                trajectory["paper_blended_velocity"],
                "c",
                "Combi--Ressler v3 Eq. (16), omitting the position-blend dW/dt term",
            )
            dataset(
                diagnostics,
                "position_time_derivative",
                trajectory["consistent_velocity"],
                "c",
                "analytic derivative of the generated coordinate centers",
            )
            dataset(
                diagnostics,
                "window_derivative",
                trajectory["window_derivative"],
                "1/M_ref",
                "analytic dW/dt with exact zero endpoint limits",
            )
            dataset(
                diagnostics,
                "discrete_velocity_error",
                trajectory["discrete_velocity_error"],
                "c",
                "norm of second-order sampled d(position)/dt minus analytic derivative",
            )
            diagnostics.attrs["max_discrete_velocity_error"] = np.float64(
                np.max(trajectory["discrete_velocity_error"])
            )

            remnant_group = handle.create_group("remnant")
            remnant_group.attrs["model"] = remnant["model"]
            remnant_group.create_dataset("mass", data=np.float64(remnant["mass"]))
            remnant_group.create_dataset("spin_chi", data=remnant["spin_chi"])
            remnant_group.create_dataset("spin_a", data=remnant["spin_a"])
            remnant_group.create_dataset("kick", data=remnant["kick"])
            if "fit" in remnant:
                fit_group = remnant_group.create_group("fit")
                for key, value in remnant["fit"].items():
                    if isinstance(value, np.ndarray):
                        fit_group.create_dataset(key, data=value)
                    else:
                        fit_group.attrs[key] = value
            if "orientation" in remnant:
                orientation_group = remnant_group.create_group("orientation")
                for key, value in remnant["orientation"].items():
                    if isinstance(value, np.ndarray):
                        orientation_group.create_dataset(key, data=value)
                    else:
                        orientation_group.attrs[key] = value
            remnant_group.create_dataset(
                "worldline_position_at_transition",
                data=trajectory["remnant_worldline_anchor"],
            )
            remnant_group["worldline_anchor"] = remnant_group[
                "worldline_position_at_transition"
            ]
            remnant_group.create_dataset(
                "transition_start", data=np.float64(trajectory["transition_start"])
            )
            remnant_group.create_dataset(
                "merger_time", data=np.float64(trajectory["merger_time"])
            )
            remnant_group.create_dataset(
                "transition_end", data=np.float64(trajectory["transition_end"])
            )
            remnant_group.create_dataset(
                "transition_separation",
                data=np.float64(
                    np.interp(
                        trajectory["transition_start"],
                        trajectory["time"],
                        trajectory["separation"],
                    )
                ),
            )

            pn = handle.create_group("pn_source")
            dataset(pn, "time", raw["time"], "M_ref", "unmodified CBwaves sample times")
            dataset(pn, "position", raw["position"], "M_ref", "unmodified PN centers")
            dataset(pn, "velocity", raw["velocity"], "c", raw["velocity_source"])
            dataset(pn, "spin_chi", raw["spin_chi"], "dimensionless", "CBwaves spins")
            dataset(pn, "separation", raw["separation"], "M_ref", "PN separation")
            if "orbital_angular_momentum" in raw:
                dataset(
                    pn,
                    "orbital_angular_momentum",
                    raw["orbital_angular_momentum"],
                    "CBwaves native",
                    "upstream Lx,Ly,Lz diagnostic",
                )

            provenance_group = handle.create_group("provenance")
            for key, value in provenance.items():
                provenance_group.attrs[key] = value
            provenance_group.attrs["dat_sha256"] = raw["dat_sha256"]
            provenance_group.attrs["correction_mask"] = pn_terms_metadata
            provenance_group.attrs["rr_parameters_json"] = canonical_json(
                rr_parameters(args.mass1_fraction, args.mass2_fraction)
            )
            provenance_group.attrs["arguments_json"] = canonical_json(
                serializable_arguments(args)
            )
            provenance_group.attrs["generator_sha256"] = generator_sha256
            if "fit" in remnant:
                provenance_group.attrs["remnant_fit_version"] = REMNANT_FIT_VERSION
                provenance_group.attrs[
                    "remnant_fit_reference_precession_sha256"
                ] = PRECESSION_REFERENCE_SHA256
            provenance_group.attrs["python_implementation"] = (
                platform.python_implementation()
            )
            provenance_group.attrs["python_version"] = platform.python_version()
            provenance_group.attrs["numpy_version"] = np.__version__
            provenance_group.attrs["h5py_version"] = h5py.__version__
            text_type = h5py.string_dtype(encoding="utf-8")
            provenance_group.create_dataset("cbwaves_ini", data=ini_text, dtype=text_type)
            handle.flush()
        if args.force:
            os.replace(temporary, output)
        else:
            try:
                # The temporary file is in output.parent, so hard-linking it
                # publishes atomically without replacing a concurrent writer.
                os.link(temporary, output)
            except FileExistsError as error:
                raise ValueError(
                    f"output already exists: {output}; pass --force to replace it"
                ) from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def serializable_arguments(args: argparse.Namespace) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return {key: convert(value) for key, value in vars(args).items()}


def validate_common_arguments(args: argparse.Namespace) -> None:
    values = (
        args.mass1_fraction,
        args.mass2_fraction,
        args.reference_mass_solar,
        args.dat_total_mass_solar,
        args.initial_separation_M,
        args.rmin_M,
        args.rmax_M,
        args.dt_M,
        args.tmax_M,
        args.post_merger_duration_M,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("masses, distances, timesteps, and durations must be positive")
    if abs(args.mass1_fraction + args.mass2_fraction - 1.0) > 1.0e-12:
        raise ValueError("--mass1-fraction and --mass2-fraction must sum to one")
    if args.mass1_fraction < args.mass2_fraction:
        raise ValueError(
            "canonical labeling requires mass1 >= mass2; swap both masses and "
            "their associated --spin1/--spin2 vectors"
        )
    if not args.rmin_M < args.initial_separation_M < args.rmax_M:
        raise ValueError("require rmin < initial separation < rmax")
    if not math.isfinite(args.eccentricity) or not 0.0 <= args.eccentricity < 1.0:
        raise ValueError("--eccentricity must lie in [0,1)")
    if not math.isfinite(args.orbits_max) or args.orbits_max <= 0.0:
        raise ValueError("--orbits-max must be positive and finite")
    if not 0 <= args.log_level <= 6:
        raise ValueError("--log-level must lie in [0,6]")
    for label, spin in (("spin1", args.spin1), ("spin2", args.spin2)):
        if len(spin) != 3 or not all(math.isfinite(value) for value in spin):
            raise ValueError(f"--{label} must contain three finite values")
        if math.sqrt(sum(value * value for value in spin)) > 1.0 + 1.0e-12:
            raise ValueError(f"--{label} must have norm <= 1")
    if args.build_jobs < 1 or args.print_step < 1:
        raise ValueError("--build-jobs and --print-step must be >= 1")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds < 0.0:
        raise ValueError("--timeout-seconds must be finite and nonnegative")
    if (
        not math.isfinite(args.transition_half_width_M)
        or args.transition_half_width_M <= 0.0
    ):
        raise ValueError("--transition-half-width-M must be positive and finite")
    if not 0.0 < args.transition_max_speed < 1.0:
        raise ValueError("--transition-max-speed must lie strictly between zero and one")
    if args.output_dt_M is not None and (
        not math.isfinite(args.output_dt_M) or args.output_dt_M <= 0.0
    ):
        raise ValueError("--output-dt-M must be positive and finite")
    if args.cbwaves_source is not None and args.convert_existing_dat is not None:
        raise ValueError("choose either --cbwaves-source or --convert-existing-dat")
    if args.convert_existing_dat is not None and not args.assume_cbwaves_harmonic:
        raise ValueError(
            "--convert-existing-dat requires --assume-cbwaves-harmonic to "
            "assert that the supplied columns, units, masses, spins and "
            "harmonic-coordinate worldlines have CBwaves semantics"
        )
    if args.remnant_fit_time_M is not None and not math.isfinite(
        args.remnant_fit_time_M
    ):
        raise ValueError("--remnant-fit-time-M must be finite")
    if args.remnant_model == "paper-nr-fit":
        if args.remnant_kick_phase_rad is None or not math.isfinite(
            args.remnant_kick_phase_rad
        ):
            raise ValueError(
                "--remnant-model=paper-nr-fit requires an explicit finite "
                "--remnant-kick-phase-rad; no random phase is generated"
            )
        if (
            args.remnant_mass is not None
            or args.remnant_spin_chi is not None
            or args.remnant_kick is not None
        ):
            raise ValueError(
                "paper-nr-fit cannot be combined with explicit remnant mass/spin/kick"
            )
    elif args.remnant_fit_time_M is not None or args.remnant_kick_phase_rad is not None:
        raise ValueError(
            "--remnant-fit-time-M and --remnant-kick-phase-rad require "
            "--remnant-model=paper-nr-fit"
        )
    if args.cbwaves_source is not None and args.dat_columns.strip().lower() != "auto":
        if parse_columns(args.dat_columns) != DAT_COLUMNS:
            raise ValueError(
                "a generated CBwaves run has a fixed extended DAT layout; use "
                "--dat-columns=auto"
            )
    if not args.dry_run and args.output is None:
        raise ValueError("--output is required unless --dry-run is used")
    if not args.dry_run and args.cbwaves_source is None and args.convert_existing_dat is None:
        raise ValueError("provide --cbwaves-source or --convert-existing-dat")
    if args.output is not None:
        output = args.output.resolve()
        if (
            args.convert_existing_dat is not None
            and output == args.convert_existing_dat.resolve()
        ):
            raise ValueError("--output must not overwrite --convert-existing-dat")
        if (
            args.cbwaves_source is not None
            and args.cbwaves_source.is_file()
            and output == args.cbwaves_source.resolve()
        ):
            raise ValueError("--output must not overwrite the CBwaves source archive")
        if not args.dry_run and output.exists() and not args.force:
            raise ValueError(
                f"output already exists: {output}; pass --force to replace it"
            )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a verified, one-line-patched Combi-Ressler CBwaves source tree "
            "or convert an existing DAT file into a KPolaris SKS HDF5 trajectory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_argument_group("source and output")
    source.add_argument(
        "--cbwaves-source",
        type=Path,
        help="official public_repo.zip or directory containing the verified CBwaves tree",
    )
    source.add_argument(
        "--convert-existing-dat",
        type=Path,
        help="skip source/build/run and convert this existing CBwaves text table",
    )
    source.add_argument(
        "--assume-cbwaves-harmonic",
        action="store_true",
        help=(
            "required caller assertion for --convert-existing-dat: selected "
            "columns, units, component labels and worldlines have CBwaves "
            "harmonic-coordinate semantics; provenance remains unverified"
        ),
    )
    source.add_argument("--output", type=Path, help="output KPolaris native HDF5 file")
    source.add_argument(
        "--force",
        action="store_true",
        help="replace an existing --output; the default is fail-if-exists",
    )
    source.add_argument("--dry-run", action="store_true", help="validate options and print the plan/INI")
    source.add_argument(
        "--dat-columns",
        default="auto",
        help=(
            "comma-separated DAT column names, or auto for this generator and "
            "the official archived wrapper layouts"
        ),
    )
    source.add_argument(
        "--dat-total-mass-solar",
        type=float,
        default=1.0,
        help="total mass used to produce --convert-existing-dat",
    )
    source.add_argument("--work-directory", type=Path, help="new directory to preserve build/run files")
    source.add_argument("--keep-work-directory", action="store_true")
    source.add_argument("--cmake", default="cmake")
    source.add_argument("--boost-root", type=Path, help="Boost prefix if CMake cannot discover it")
    source.add_argument(
        "--cmake-prefix-path",
        type=Path,
        action="append",
        default=[],
        help="additional CMake package prefix; repeat as needed",
    )
    source.add_argument("--build-jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    source.add_argument("--timeout-seconds", type=float, default=0.0)

    binary = parser.add_argument_group("binary and CBwaves integration")
    binary.add_argument("--mass1-fraction", type=float, default=0.5)
    binary.add_argument("--mass2-fraction", type=float, default=0.5)
    binary.add_argument("--reference-mass-solar", type=float, default=1.0)
    binary.add_argument("--spin1", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 0.0))
    binary.add_argument("--spin2", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 0.0))
    binary.add_argument("--initial-separation-M", type=float, default=10.0)
    binary.add_argument("--rmin-M", type=float, default=3.0)
    binary.add_argument("--rmax-M", type=float, default=150.0)
    binary.add_argument("--dt-M", type=float, default=0.01)
    binary.add_argument("--tmax-M", type=float, default=2000.0)
    binary.add_argument("--orbits-max", type=float, default=1.0e6)
    binary.add_argument("--eccentricity", type=float, default=0.0)
    binary.add_argument("--print-step", type=int, default=1)
    binary.add_argument("--log-level", type=int, default=3)
    binary.add_argument(
        "--allow-truncated-inspiral",
        action="store_true",
        help=(
            "allow the PN state selected at merger time to remain above rmin "
            "(diagnostic/nonphysical runs only)"
        ),
    )

    transition = parser.add_argument_group("v3 merger transition")
    transition.add_argument("--merger-time-M", type=float)
    transition.add_argument("--transition-start-M", type=float)
    transition.add_argument("--transition-end-M", type=float)
    transition.add_argument(
        "--transition-half-width-M",
        type=float,
        default=0.01,
        help=(
            "initial half-width when explicit endpoints are omitted; the "
            "consistent mode widens it automatically if needed"
        ),
    )
    transition.add_argument(
        "--transition-velocity",
        choices=("consistent", "paper-blend"),
        default="consistent",
        help=(
            "consistent uses ds/dt for the boost; paper-blend reproduces the "
            "independent velocity interpolation in paper v3 Eq. (16)"
        ),
    )
    transition.add_argument(
        "--auto-expand-transition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="widen an implicit consistent transition to keep its boost subluminal",
    )
    transition.add_argument(
        "--transition-max-speed",
        type=float,
        default=0.95,
        help="maximum accepted boost speed in consistent mode",
    )
    transition.add_argument("--post-merger-duration-M", type=float, default=0.1)
    transition.add_argument("--output-dt-M", type=float)
    transition.add_argument(
        "--remnant-model",
        choices=("equal-nonspinning", "paper-nr-fit", "explicit"),
        default="equal-nonspinning",
    )
    transition.add_argument(
        "--remnant-fit-time-M",
        type=float,
        help=(
            "paper-nr-fit input time; uses the latest recorded CBwaves sample "
            "at or before this time (default: the actual transition start, "
            "t_merger-dt_buffer, as prescribed in the paper)"
        ),
    )
    transition.add_argument(
        "--remnant-kick-phase-rad",
        type=float,
        help=(
            "required deterministic merger phase Theta for paper-nr-fit's "
            "out-of-plane kick term; radians, never sampled randomly"
        ),
    )
    transition.add_argument("--remnant-mass", type=float)
    transition.add_argument(
        "--remnant-spin-chi", type=float, nargs=3, metavar=("X", "Y", "Z")
    )
    transition.add_argument(
        "--remnant-kick", type=float, nargs=3, metavar=("VX", "VY", "VZ")
    )
    return parser


def dry_run_plan(args: argparse.Namespace, ini_text: str) -> dict[str, object]:
    validation: dict[str, object] = (
        {"status": "unverified external DAT; no paper/source identity asserted"}
        if args.convert_existing_dat is not None
        else {"status": "deferred (no source supplied)"}
    )
    if args.cbwaves_source is not None:
        if args.cbwaves_source.is_file():
            actual = sha256_file(args.cbwaves_source)
            if actual != ZENODO_ARCHIVE_SHA256:
                raise ValueError(
                    f"official archive hash mismatch: expected {ZENODO_ARCHIVE_SHA256}, got {actual}"
                )
            validation = {"status": "verified official archive", "sha256": actual}
        else:
            validation = {
                "status": "verified official source tree",
                "files": verify_cbwaves_tree(args.cbwaves_source),
            }
    return {
        "action": "dry-run",
        "schema": f"{SCHEMA_NAME}/{SCHEMA_VERSION}",
        "source_validation": validation,
        "temporary_patch": (
            None if args.convert_existing_dat is not None
            else "9130111./3306. -> 9130111./3360."
        ),
        "correction_mask": (
            None if args.convert_existing_dat is not None else list(PN_TERMS)
        ),
        "transition_velocity": args.transition_velocity,
        "remnant_model": args.remnant_model,
        "remnant_fit_status": (
            "deferred until the CBwaves table is loaded"
            if args.remnant_model in ("paper-nr-fit", "equal-nonspinning")
            else "resolved from command-line inputs"
        ),
        "auto_expand_transition": args.auto_expand_transition,
        "output": None if args.output is None else str(args.output),
        "convert_existing_dat": (
            None if args.convert_existing_dat is None else str(args.convert_existing_dat)
        ),
        "cbwaves_ini": ini_text,
    }


def execute(args: argparse.Namespace) -> Path | None:
    validate_common_arguments(args)
    # Resolve here as well as after conversion so an invalid generic remnant
    # cannot quietly pass dry-run validation.
    if args.remnant_model != "paper-nr-fit":
        # Validate command-line-only constraints before a potentially costly
        # source build.  The equal-nonspinning direction is resolved again
        # from the loaded orbital angular momentum below.
        resolve_remnant(args)
    ini_text = render_cbwaves_ini(args)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args, ini_text), indent=2, sort_keys=True))
        return None

    provenance: dict[str, str]
    cleanup = False
    work_root: Path | None = None
    try:
        if args.convert_existing_dat is not None:
            dat_path = args.convert_existing_dat.resolve()
            total_mass_solar = args.dat_total_mass_solar
            provenance = {
                "source_verification": "unverified-external-dat",
                "archive_sha256": "not-asserted",
                "upstream_cbwaves_sha256": "not-asserted",
                "patched_cbwaves_sha256": "not-asserted",
                "patch": "not-asserted; no source was supplied or modified",
            }
            ini_text = "# Existing DAT conversion: no INI was executed by this run.\n"
        else:
            if args.work_directory is not None:
                work_root = args.work_directory.resolve()
                if work_root.exists():
                    raise ValueError("--work-directory must not already exist")
                work_root.mkdir(parents=True)
            else:
                work_root = Path(tempfile.mkdtemp(prefix="kpolaris-cbwaves-"))
                cleanup = not args.keep_work_directory
            assert args.cbwaves_source is not None
            patched_source, provenance = prepare_patched_source(
                args.cbwaves_source.resolve(), work_root
            )
            binary = build_cbwaves(
                patched_source,
                work_root,
                args.cmake,
                args.build_jobs,
                args.boost_root,
                args.cmake_prefix_path,
            )
            run_root = work_root / "run"
            run_root.mkdir()
            ini_path = run_root / "trajectory.ini"
            ini_path.write_text(ini_text, encoding="utf-8")
            timeout = None if args.timeout_seconds <= 0.0 else args.timeout_seconds
            run_checked([str(binary), str(ini_path)], cwd=run_root, timeout_seconds=timeout)
            dat_path = run_root / "trajectory.dat"
            if not dat_path.is_file() or dat_path.stat().st_size == 0:
                raise ValueError("CBwaves did not produce trajectory.dat")
            total_mass_solar = args.reference_mass_solar

        raw = load_cbwaves_dat(
            dat_path,
            columns=resolve_dat_columns(dat_path, args.dat_columns),
            total_mass_solar=total_mass_solar,
            mass1=args.mass1_fraction,
            mass2=args.mass2_fraction,
        )
        validate_raw_remnant_compatibility(args, raw)
        raw["merger_reach"] = validate_merger_reached(raw, args)
        raw["trajectory_status"] = (
            "merger_separation_reached"
            if raw["merger_reach"]["reached"]
            else "diagnostic_truncated"
        )
        remnant, trajectory = resolve_remnant_and_trajectory(args, raw)
        assert args.output is not None
        write_native_hdf5(
            args.output,
            raw=raw,
            trajectory=trajectory,
            remnant=remnant,
            args=args,
            provenance=provenance,
            ini_text=ini_text,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "samples": len(trajectory["time"]),
                    "source_samples": len(raw["time"]),
                    "merger_separation_reached": raw["merger_reach"]["reached"],
                    "trajectory_status": raw["trajectory_status"],
                    "final_pn_separation_M": raw["merger_reach"][
                        "final_separation"
                    ],
                    "selected_merger_separation_M": raw["merger_reach"][
                        "selected_merger_separation"
                    ],
                    "merger_reach_evaluation_time_M": raw["merger_reach"][
                        "evaluation_time"
                    ],
                    "transition_start_M": trajectory["transition_start"],
                    "requested_transition_start_M": trajectory[
                        "requested_transition_start"
                    ],
                    "merger_time_M": trajectory["merger_time"],
                    "transition_end_M": trajectory["transition_end"],
                    "requested_transition_end_M": trajectory[
                        "requested_transition_end"
                    ],
                    "transition_auto_expansions": trajectory[
                        "transition_auto_expansions"
                    ],
                    "transition_velocity": args.transition_velocity,
                    "remnant_model": args.remnant_model,
                    "remnant_mass": remnant["mass"],
                    "remnant_spin_chi": remnant["spin_chi"].tolist(),
                    "remnant_kick": remnant["kick"].tolist(),
                    "worldline_velocity_consistent": trajectory[
                        "worldline_velocity_consistent"
                    ],
                    "maximum_boost_speed": trajectory["maximum_speed"],
                    "exact_remnant_from_M": trajectory["transition_end"],
                    "work_directory": None if work_root is None else str(work_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return args.output.resolve()
    finally:
        if cleanup and work_root is not None:
            shutil.rmtree(work_root)


def main(argv: Iterable[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        execute(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
