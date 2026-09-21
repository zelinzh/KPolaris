#!/usr/bin/env python3

"""Build core tests from a staged source tree with no Git metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


REVISION_SENTINEL = "archive-contract-revision"


def run(
    command: list[str], cwd: Path, *, expect_ok: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expect_ok and result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}"
        )
    if not expect_ok and result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {' '.join(command)}")
    return result


def git_source_files(source_dir: Path) -> list[Path] | None:
    if not (source_dir / ".git").exists() or shutil.which("git") is None:
        return None
    result = subprocess.run(
        [
            "git",
            "-C",
            str(source_dir),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return [
        Path(item.decode("utf-8"))
        for item in result.stdout.split(b"\0")
        if item
    ]


def fallback_source_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    skipped_roots = {".git", ".spack-env", "validation", "runs", "outputs", "__pycache__"}
    for directory, children, names in os.walk(source_dir):
        children[:] = [name for name in children
                       if name not in skipped_roots and not name.startswith(("build", "cmake-build"))]
        for name in names:
            path = Path(directory) / name
            if path.suffix != ".pyc" and path.is_file():
                files.append(path.relative_to(source_dir))
    return files


def stage_source(source_dir: Path, stage_dir: Path) -> int:
    source_files = git_source_files(source_dir)
    if source_files is None:
        source_files = fallback_source_files(source_dir)
    required = {
        Path("CMakeLists.txt"),
        Path("CITATION.cff"),
        Path("cmake/KPolarisConfig.cmake.in"),
        Path("cmake/kpolaris_version.hpp.in"),
        Path("tests/test_core.cpp"),
    }
    missing = required.difference(source_files)
    if missing:
        raise AssertionError(f"source export omits required files: {sorted(map(str, missing))}")

    for relative in source_files:
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError(f"unsafe source path: {relative}")
        source = source_dir / relative
        destination = stage_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return len(source_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--make-program", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--kokkos-dir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    stage_dir = workdir / "KPolaris-source"
    build_dir = workdir / "build"
    stage_dir.mkdir(parents=True)
    copied = stage_source(source_dir, stage_dir)

    if (stage_dir / ".git").exists():
        raise AssertionError("staged source unexpectedly contains .git metadata")
    # The staged tree lives below the development checkout's build directory.
    # Prevent Git (including CMake's provenance probe) from walking upward into
    # that checkout; this makes the test equivalent to an extracted archive.
    os.environ["GIT_CEILING_DIRECTORIES"] = str(workdir)
    if shutil.which("git") is not None:
        run(["git", "-C", str(stage_dir), "rev-parse", "HEAD"], workdir, expect_ok=False)

    def configure_command(target: Path, build_tests: bool) -> list[str]:
        return [
            args.cmake,
            "-S",
            str(stage_dir),
            "-B",
            str(target),
            "-G",
            args.generator,
            f"-DCMAKE_MAKE_PROGRAM={args.make_program}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={args.cxx_compiler}",
            "-DKPOLARIS_FETCH_KOKKOS=OFF",
            f"-DKokkos_DIR={args.kokkos_dir.resolve()}",
            "-DKPOLARIS_BUILD_IMAGE_TOOL=OFF",
            "-DKPOLARIS_BUILD_TRACE_TOOL=OFF",
            "-DKPOLARIS_BUILD_EXAMPLES=OFF",
            "-DKPOLARIS_BUILD_AUX_TOOLS=OFF",
            f"-DKPOLARIS_BUILD_TESTS={'ON' if build_tests else 'OFF'}",
        ]

    auto_build_dir = workdir / "build-auto-provenance"
    run(configure_command(auto_build_dir, False), workdir)
    auto_version_header = (
        auto_build_dir / "generated" / "common" / "version.hpp"
    ).read_text(encoding="utf-8")
    for marker in (
        'inline constexpr const char* source_revision = "unknown";',
        "inline constexpr int source_dirty = -1;",
    ):
        if marker not in auto_version_header:
            raise AssertionError(
                f"automatic no-Git provenance header lacks {marker!r}"
            )

    configure = configure_command(build_dir, True)
    configure.extend(
        [
            "-DKPOLARIS_ENABLE_WARNINGS=ON",
            f"-DKPOLARIS_SOURCE_REVISION={REVISION_SENTINEL}",
            "-DKPOLARIS_SOURCE_DIRTY=OFF",
        ]
    )
    run(configure, workdir)
    run(
        [
            args.cmake,
            "--build",
            str(build_dir),
            "--target",
            "kpolaris_unit_tests",
            "kpolaris_numerical_validation",
            "-j2",
        ],
        workdir,
    )
    run([str(build_dir / "kpolaris_unit_tests")], workdir)
    run([str(build_dir / "kpolaris_numerical_validation")], workdir)

    version_header = (build_dir / "generated" / "common" / "version.hpp").read_text(
        encoding="utf-8"
    )
    for marker in (
        f'inline constexpr const char* version = "{args.expected_version}";',
        f'inline constexpr const char* source_revision = "{REVISION_SENTINEL}";',
        "inline constexpr int source_dirty = 0;",
        'inline constexpr const char* build_type = "Release";',
    ):
        if marker not in version_header:
            raise AssertionError(f"generated archive version header lacks {marker!r}")

    print(
        f"verified no-Git source build: {copied} files, auto provenance is "
        "unknown, revision override and clean source flag embedded, "
        "unit/numerical tests passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
