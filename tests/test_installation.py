#!/usr/bin/env python3

"""Exercise the installed runtime and the exported header-only CMake target."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


PARAMETER_FILES = {
    "athenak_recommended.par",
    "binary_riaf.par",
    "demo_riaf.par",
    "iharm_recommended.par",
    "iharm_sks_precomputed.par",
    "iharm_sks_primitives.par",
    "kharma_recommended.par",
    "riaf_recommended.par",
    "torus_recommended.par",
}

ANALYSIS_SCRIPTS = {
    "analyze_closure_stokes.py",
    "analyze_physical_responses.py",
    "analyze_polarization_budget.py",
    "analyze_trace_mechanisms.py",
    "benchmark_image_scaling.py",
    "bin_trace_physical_maps.py",
    "compare_analysis_trace.py",
    "compare_slow_light_batches.py",
    "compare_stokes_csv.py",
    "compare_stokes_hdf5.py",
    "export_wiki.py",
    "generate_binary_trajectory.py",
    "kpolaris.py",
    "kpolaris_ddc_parallel.py",
    "kpolaris_ddc_reconstruct.py",
    "kpolaris_evpa.py",
    "kpolaris_image.py",
    "plot_analysis_maps.py",
    "plot_direct_only_compare.py",
    "plot_equatorial_compare.py",
    "plot_kpolaris_pol.py",
    "plot_physical_response_gallery.py",
    "plot_stokes_compare.py",
    "plot_trace_diagnostics.py",
    "prepare_kharma_cache.py",
    "serve_ddc_parallel.py",
}

DOCUMENT_FILES = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "README_zh.md",
}


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}"
        )
    return result


def executable_path(directory: Path, name: str) -> Path:
    candidate = directory / name
    if candidate.is_file():
        return candidate
    if os.name == "nt" and not name.lower().endswith(".exe"):
        candidate = directory / f"{name}.exe"
        if candidate.is_file():
            return candidate
    raise AssertionError(f"installed executable is missing: {directory / name}")


def require_exact_files(directory: Path, expected: set[str], pattern: str) -> None:
    actual = {path.name for path in directory.glob(pattern) if path.is_file()}
    if actual != expected:
        raise AssertionError(
            f"unexpected files in {directory}: expected {sorted(expected)}, got {sorted(actual)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--make-program", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--kokkos-dir", type=Path, required=True)
    parser.add_argument("--kokkos-compile-launcher", type=Path)
    parser.add_argument("--kokkos-nvcc-wrapper", type=Path)
    parser.add_argument("--bindir", type=Path, required=True)
    parser.add_argument("--includedir", type=Path, required=True)
    parser.add_argument("--libdir", type=Path, required=True)
    parser.add_argument("--libexecdir", type=Path, required=True)
    parser.add_argument("--datadir", type=Path, required=True)
    parser.add_argument(
        "--runtime-library-dir", type=Path, action="append", default=[]
    )
    parser.add_argument("--runtime-name", action="append", default=[])
    args = parser.parse_args()

    # Installed runtimes intentionally keep HDF5/CUDA/Kokkos as external
    # dependencies.  Exercise the relocated KPolaris tree with the dependency
    # directories discovered by its build. The exported header-only CMake target
    # does not encode these paths; CUDA runtimes also retain dependency RPATHs.
    runtime_library_dirs = [
        str(path.resolve()) for path in args.runtime_library_dir if path.is_dir()
    ]
    if runtime_library_dirs:
        if os.name == "nt":
            loader_path_name = "PATH"
        elif sys.platform == "darwin":
            loader_path_name = "DYLD_LIBRARY_PATH"
        else:
            loader_path_name = "LD_LIBRARY_PATH"
        existing_loader_path = os.environ.get(loader_path_name, "")
        loader_paths = runtime_library_dirs
        if existing_loader_path:
            loader_paths.append(existing_loader_path)
        os.environ[loader_path_name] = os.pathsep.join(loader_paths)

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    install_prefix = workdir / "prefix"

    install_command = [
        args.cmake,
        "--install",
        str(args.build_dir.resolve()),
        "--prefix",
        str(install_prefix),
        "--config",
        args.config,
    ]
    run(install_command, workdir)
    prefix = workdir / "relocated-prefix"
    shutil.move(str(install_prefix), str(prefix))
    if install_prefix.exists():
        raise AssertionError("installed tree was not moved for relocation testing")

    runtime_dir = prefix / args.bindir
    expected_runtimes = set(args.runtime_name)
    actual_runtimes = (
        {path.name for path in runtime_dir.iterdir() if path.is_file()}
        if runtime_dir.is_dir()
        else set()
    )
    if actual_runtimes != expected_runtimes:
        raise AssertionError(
            "installed runtime directory contains unexpected files: "
            f"expected {sorted(expected_runtimes)}, got {sorted(actual_runtimes)}"
        )
    for name in sorted(expected_runtimes):
        executable = executable_path(runtime_dir, name)
        if not os.access(executable, os.X_OK):
            raise AssertionError(f"installed runtime is not executable: {executable}")
        version = run([str(executable), "--version"], workdir).stdout
        for marker in (
            f"KPolaris {args.expected_version}",
            "revision ",
            "source_dirty ",
            "compiler ",
            "build_type ",
        ):
            if marker not in version:
                raise AssertionError(f"{executable} --version lacks {marker!r}:\n{version}")

    require_exact_files(
        prefix / args.datadir / "kpolaris" / "params", PARAMETER_FILES, "*.par"
    )
    if "kpolaris_model_image_riaf" in expected_runtimes:
        smoke_output = workdir / "installed-riaf-smoke.h5"
        smoke = run(
            [
                str(executable_path(runtime_dir, "kpolaris_model_image_riaf")),
                "--parameter_file="
                + str(
                    prefix
                    / args.datadir
                    / "kpolaris"
                    / "params"
                    / "riaf_recommended.par"
                ),
                "--nx=2",
                "--ny=2",
                f"--output={smoke_output}",
                "--parameter_output=none",
                "--adaptive_tolerance=1e-8",
                "--timing=0",
            ],
            workdir,
        ).stdout
        if not smoke_output.is_file() or smoke_output.stat().st_size == 0:
            raise AssertionError("relocated installed RIAF runtime did not write HDF5")
        for marker in ("format hdf5", "pixels 4", "returned 4", "flux_I_Jy"):
            if marker not in smoke:
                raise AssertionError(f"installed RIAF smoke output lacks {marker!r}")
    script_dir = prefix / args.libexecdir / "kpolaris"
    require_exact_files(script_dir, ANALYSIS_SCRIPTS, "*.py")
    for script in script_dir.glob("*.py"):
        if not os.access(script, os.X_OK):
            raise AssertionError(f"installed analysis script is not executable: {script}")
    require_exact_files(
        prefix / args.datadir / "doc" / "KPolaris", DOCUMENT_FILES, "*"
    )

    include_root = prefix / args.includedir / "kpolaris"
    if (prefix / args.includedir / "kokkos").exists():
        raise AssertionError("KPolaris install unexpectedly bundled Kokkos headers")
    for header in (
        include_root / "KPolaris.hpp",
        include_root / "common" / "types.hpp",
        include_root / "common" / "version.hpp",
        include_root / "radiation" / "stokes.hpp",
        include_root / "radiation" / "emission_selection.hpp",
    ):
        if not header.is_file():
            raise AssertionError(f"installed public header is missing: {header}")
    internal_loader = include_root / "geometry" / "binary_trajectory_hdf5.hpp"
    if internal_loader.exists():
        raise AssertionError(
            "runtime-internal HDF5 loader header leaked into the header-only "
            "installed API without its compiled implementation"
        )

    cmake_package = prefix / args.libdir / "cmake" / "KPolaris"
    for cmake_file in (
        "KPolarisConfig.cmake",
        "KPolarisConfigVersion.cmake",
        "KPolarisTargets.cmake",
    ):
        if not (cmake_package / cmake_file).is_file():
            raise AssertionError(f"installed CMake package file is missing: {cmake_file}")
    exported_targets = (cmake_package / "KPolarisTargets.cmake").read_text(
        encoding="utf-8"
    )
    for forbidden in (str(args.build_dir.resolve()), "-fsanitize=", "-Wall", "/W4"):
        if forbidden in exported_targets:
            raise AssertionError(
                f"installed CMake target leaked build-only value {forbidden!r}"
            )

    consumer_source = workdir / "consumer-source"
    consumer_build = workdir / "consumer-build"
    consumer_source.mkdir()
    (consumer_source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(KPolarisInstallConsumer LANGUAGES CXX)\n"
        f"find_package(KPolaris {args.expected_version} CONFIG REQUIRED)\n"
        "add_executable(kpolaris_install_consumer main.cpp)\n"
        "target_link_libraries(kpolaris_install_consumer PRIVATE KPolaris::kpolaris)\n",
        encoding="utf-8",
    )
    (consumer_source / "main.cpp").write_text(
        "#include <iostream>\n"
        "#include \"KPolaris.hpp\"\n"
        "int main() {\n"
        "  kpolaris::Stokes<double> stokes{};\n"
        "  if (kpolaris::ndim != 4 || stokes.I != 0.0) return 1;\n"
        "  std::cout << \"consumer-ok\\n\";\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )

    configure = [
        args.cmake,
        "-S",
        str(consumer_source),
        "-B",
        str(consumer_build),
        "-G",
        args.generator,
        f"-DCMAKE_MAKE_PROGRAM={args.make_program}",
        f"-DCMAKE_BUILD_TYPE={args.config}",
        f"-DCMAKE_CXX_COMPILER={args.cxx_compiler}",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DKokkos_DIR={args.kokkos_dir.resolve()}",
    ]
    if args.kokkos_compile_launcher is not None:
        launcher = args.kokkos_compile_launcher.resolve()
        if not launcher.is_file():
            raise AssertionError(f"Kokkos compile launcher is missing: {launcher}")
        configure.append(f"-DKokkos_COMPILE_LAUNCHER={launcher}")
    if args.kokkos_nvcc_wrapper is not None:
        wrapper = args.kokkos_nvcc_wrapper.resolve()
        if not wrapper.is_file():
            raise AssertionError(f"Kokkos nvcc_wrapper is missing: {wrapper}")
        configure.append(f"-DKokkos_NVCC_WRAPPER={wrapper}")
    run(configure, workdir)
    run(
        [args.cmake, "--build", str(consumer_build), "--config", args.config],
        workdir,
    )
    consumer = executable_path(consumer_build, "kpolaris_install_consumer")
    if run([str(consumer)], workdir).stdout.strip() != "consumer-ok":
        raise AssertionError("installed package consumer returned unexpected output")

    print(
        f"verified install: {len(set(args.runtime_name))} runtimes, "
        f"{len(PARAMETER_FILES)} parameter files, downstream CMake consumer passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
