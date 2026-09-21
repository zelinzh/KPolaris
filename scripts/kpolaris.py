#!/usr/bin/env python3
"""KPolaris environment checks, selective builds, demos and image inspection.

This entry point uses the Python standard library until an image is inspected.
It never installs system packages or changes the shell environment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("riaf", "torus", "iharm", "kharma", "athenak", "bhac", "hamr", "binary_riaf")
EXAMPLES = ("image", "multifrequency", "diagnostics", "response")
ARCHITECTURES = {"7.0": "VOLTA70", "7.5": "TURING75", "8.0": "AMPERE80",
                 "8.6": "AMPERE86", "8.9": "ADA89", "9.0": "HOPPER90", "12.0": "BLACKWELL120"}


def probe(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def cache_values(directory):
    path = directory / "CMakeCache.txt"
    if not path.is_file():
        return {}
    return {m[1]: m[2] for line in path.read_text().splitlines()
            if (m := re.match(r"([^/#][^:]*):[^=]+=(.*)$", line))}


def environment(backend="cpu", build_dir=None):
    checks = []
    def check(name, ok, detail, required=True):
        checks.append({"name": name, "status": "ok" if ok else "missing",
                       "required": required, "detail": detail})
    check("python", sys.version_info >= (3, 11), "Python 3.11+; running " + sys.version.split()[0])
    cmake = shutil.which("cmake")
    _, version, _ = probe([cmake, "--version"]) if cmake else (1, "", "")
    match = re.search(r"version (\d+)\.(\d+)\.(\d+)", version)
    check("cmake", bool(match and tuple(map(int, match.groups())) >= (3, 25, 0)),
          version.splitlines()[0] if version else "CMake 3.25+ is required")
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    check("cxx", bool(compiler and shutil.which(compiler)), compiler or "a C++20 compiler is required; CMake verifies language support")
    check("build_tool", bool(shutil.which("ninja") or shutil.which("make")), "Ninja or Make")
    check("hdf5_hint", bool(shutil.which("h5c++") or os.environ.get("HDF5_ROOT")),
          "HDF5 C++ is verified by CMake; activate your dependency environment or set HDF5_ROOT", False)
    for module in ("numpy", "h5py", "matplotlib"):
        code, out, err = probe([sys.executable, "-c", f"import {module}; print({module}.__version__)"])
        check(module, code == 0, out if code == 0 else f"install {module} in {sys.executable}")
    gpus = []
    smi = shutil.which("nvidia-smi")
    if smi:
        code, out, err = probe([smi, "--query-gpu=index,name,compute_cap,memory.total", "--format=csv,noheader,nounits"])
        if code == 0:
            for line in out.splitlines():
                fields = [v.strip() for v in line.split(",")]
                if len(fields) == 4:
                    gpus.append({"index": fields[0], "name": fields[1], "compute_capability": fields[2],
                                 "memory_mib": fields[3], "kokkos_arch": ARCHITECTURES.get(fields[2])})
    if backend == "cuda":
        check("nvcc", bool(shutil.which("nvcc")), shutil.which("nvcc") or "activate a CUDA toolkit environment")
        check("nvidia_driver", bool(gpus), "a visible NVIDIA GPU is required to run the CUDA demo")
    cache = cache_values(build_dir) if build_dir else {}
    return {"schema": "kpolaris.environment.v1", "ok": all(c["status"] == "ok" for c in checks if c["required"]),
            "backend_requested": backend, "python": sys.executable, "cpu_threads": os.cpu_count(),
            "checks": checks, "gpus": gpus,
            "build": {"directory": str(build_dir) if build_dir else None,
                      "source": cache.get("CMAKE_HOME_DIRECTORY"),
                      "models": cache.get("KPOLARIS_IMAGE_MODELS"),
                      "cuda": cache.get("KPOLARIS_ENABLE_CUDA"), "openmp": cache.get("KPOLARIS_ENABLE_OPENMP"),
                      "executables": sorted(p.name for p in build_dir.glob("kpolaris_model_image_*")) if build_dir and build_dir.is_dir() else []}}


def emit(payload, as_json):
    if as_json:
        print(json.dumps(payload, indent=2, allow_nan=False))
    elif payload.get("schema") == "kpolaris.environment.v1":
        print(f"KPolaris environment ({payload['backend_requested']})")
        for check in payload["checks"]:
            state = check["status"] if check["required"] else "info"
            print(f"  {state:7} {check['name']}: {check['detail']}")
        for gpu in payload["gpus"]:
            print(f"  GPU {gpu['index']}: {gpu['name']}, {gpu['memory_mib']} MiB, architecture {gpu['kokkos_arch'] or 'specify explicitly'}")
        print("Ready for the build/demo checks." if payload["ok"] else "Install/activate the missing dependencies, then run doctor again.")
    else:
        print(json.dumps(payload, indent=2, allow_nan=False))


def choose_arch(args):
    if args.cuda_arch:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", args.cuda_arch):
            raise ValueError("cuda-arch must be a Kokkos architecture name, such as AMPERE80")
        return args.cuda_arch
    caps = {g["kokkos_arch"] for g in environment("cuda")["gpus"]}
    if len(caps) == 1 and None not in caps:
        return caps.pop()
    raise ValueError("specify --cuda-arch for this machine (no unique supported GPU architecture detected)")


def build_plan(args, root, build):
    models = [v for v in args.models.split(",") if v]
    if not models or any(v not in MODELS for v in models):
        raise ValueError("models must be a comma-separated subset of " + ",".join(MODELS))
    cache = cache_values(build)
    if cache:
        if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != root:
            raise ValueError("build directory belongs to another source tree; choose a new --build-dir")
        old_backend = "cuda" if cache.get("KPOLARIS_ENABLE_CUDA") == "ON" else "openmp" if cache.get("KPOLARIS_ENABLE_OPENMP") == "ON" else "cpu"
        if old_backend != args.backend:
            raise ValueError("use separate build directories for CPU, OpenMP and CUDA")
    base = ["cmake", "-S", str(root), "-B", str(build)]
    if not cache and shutil.which("ninja"):
        base += ["-G", "Ninja"]
    base += ["-DCMAKE_BUILD_TYPE=Release", "-DKPOLARIS_BUILD_EXAMPLES=OFF", "-DKPOLARIS_BUILD_AUX_TOOLS=OFF",
             "-DKPOLARIS_BUILD_IMAGE_TOOL=ON", f"-DKPOLARIS_BUILD_TESTS={'ON' if args.tests else 'OFF'}",
             f"-DKPOLARIS_BUILD_TRACE_TOOL={'ON' if args.trace else 'OFF'}", "-DKPOLARIS_ENABLE_ANALYSIS_MODE=ON",
             "-DKPOLARIS_MAX_FREQUENCIES=2", f"-DKPOLARIS_IMAGE_MODELS={';'.join(models)}",
             f"-DKPOLARIS_GRMHD_COORDINATES={args.coordinates}",
             f"-DKPOLARIS_ENABLE_CUDA={'ON' if args.backend == 'cuda' else 'OFF'}",
             f"-DKPOLARIS_ENABLE_OPENMP={'ON' if args.backend == 'openmp' else 'OFF'}",
             f"-DKPOLARIS_ENABLE_SERIAL={'OFF' if args.backend == 'openmp' else 'ON'}",
             f"-DPython3_EXECUTABLE={sys.executable}"]
    steps = []
    if args.kokkos_dir:
        base += ["-DKPOLARIS_FETCH_KOKKOS=OFF", f"-DKokkos_DIR={args.kokkos_dir.resolve()}"]
    else:
        base += ["-DKPOLARIS_FETCH_KOKKOS=ON"]
    if args.backend == "cuda":
        arch = choose_arch(args)
        if cache.get("KPOLARIS_CUDA_ARCH", arch) != arch:
            raise ValueError("use a new build directory when changing GPU architecture")
        wrapper = args.nvcc_wrapper
        if not wrapper:
            if args.kokkos_dir:
                raise ValueError("external CUDA Kokkos requires --nvcc-wrapper from the same installation")
            bootstrap = build.with_name(build.name + "-bootstrap")
            steps.append(("bootstrap", ["cmake", "-S", str(root), "-B", str(bootstrap),
                         "-DKPOLARIS_BUILD_IMAGE_TOOL=OFF", "-DKPOLARIS_BUILD_TRACE_TOOL=OFF",
                         "-DKPOLARIS_BUILD_TESTS=OFF", "-DKPOLARIS_BUILD_EXAMPLES=OFF",
                         "-DKPOLARIS_BUILD_AUX_TOOLS=OFF", "-DKPOLARIS_ENABLE_CUDA=OFF", *args.cmake_arg]))
            wrapper = bootstrap / "_deps/kokkos-src/bin/nvcc_wrapper"
        base += [f"-DCMAKE_CXX_COMPILER={wrapper.resolve()}", f"-DKPOLARIS_CUDA_ARCH={arch}"]
    elif args.nvcc_wrapper:
        raise ValueError("nvcc-wrapper is only used with --backend=cuda")
    steps.append(("configure", base + args.cmake_arg))
    build_command = ["cmake", "--build", str(build), "--parallel", str(args.jobs)]
    if not args.tests and not args.trace:
        build_command += ["--target", *[f"kpolaris_model_image_{m}" for m in models]]
    steps.append(("build", build_command))
    if args.tests:
        steps.append(("tests", ["ctest", "--test-dir", str(build), "--output-on-failure"]))
    return steps


def run_steps(steps, directory, root, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    for name, command in steps:
        log = directory / f"{name}.log"
        print(f"[{name}] {log}", file=sys.stderr, flush=True)
        record = {"stage": name, "command": command, "log": str(log)}
        manifest.setdefault("commands", []).append(record)
        (directory / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with log.open("w") as handle:
            p = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        record["returncode"] = p.returncode
        (directory / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if p.returncode:
            raise RuntimeError(f"{name} exited {p.returncode}; see {log}\n" + log.read_text()[-2000:])


def require_plot_dependencies(plot=True):
    missing = [n for n in (["numpy", "h5py", "matplotlib"] if plot else ["numpy", "h5py"])
               if importlib.util.find_spec(n) is None]
    if missing:
        raise ValueError("missing Python modules: " + ", ".join(missing) + f"; install requirements.txt using {sys.executable}")


def demo_plan(args, root, build, output):
    executable = args.image_exe.resolve() if args.image_exe else build / "kpolaris_model_image_riaf"
    common = [str(executable), f"--parameter_file={root / 'params/demo_riaf.par'}",
              f"--nx={args.resolution}", f"--ny={args.resolution}", "--parameter_output=auto",
              "--kokkos-print-configuration"]
    if args.backend == "openmp":
        common += [f"--kokkos-num-threads={args.threads}"]
    jobs = []
    if args.example == "response":
        for kind in ("density_scale", "temperature_scale", "magnetic_scale"):
            jobs.append((f"riaf_{kind}", ["--analysis_mode=1", f"--analysis_response={kind}",
                        "--analysis_partition=" + ("radial" if kind == "temperature_scale" else "region")]))
    else:
        extra = {"image": [], "multifrequency": ["--freq_list=230e9,345e9"],
                 "diagnostics": ["--analysis_mode=1", "--analysis_response=none", "--analysis_partition=radial"]}[args.example]
        jobs.append((args.example, extra))
    steps, images = [], []
    for name, extra in jobs:
        path = output / f"{name}.h5"; images.append(path)
        steps.append((name, common + extra + [f"--output={path}"]))
        if not args.no_plot:
            for fi in range(2 if args.example == "multifrequency" else 1):
                presentation = (["--layout=intensity", "--intensity-unit=brightness-temperature",
                                 "--title=M87* · analytic RIAF"]
                                if args.example in ("image", "multifrequency") else [])
                steps.append((f"{name}_plot_{fi}", [sys.executable, str(root / "scripts/plot_kpolaris_pol.py"),
                              str(path), f"--freq-index={fi}", "--fov-units=muas", *presentation,
                              f"--output={output / (name + f'_freq{fi}.png')}"]))
    if not args.no_plot and args.example == "diagnostics":
        steps.append(("diagnostic_maps", [sys.executable, str(root / "scripts/plot_analysis_maps.py"), str(images[0]),
                      "--fields=observer_weighted_radius_I,intensity_formation_radius_median,absorption_depth,faraday_rotation_depth",
                      "--fov-units=M", f"--output={output / 'diagnostic_maps.png'}"]))
    if not args.no_plot and args.example == "response":
        steps.append(("responses", [sys.executable, str(root / "scripts/plot_physical_response_gallery.py"), str(output),
                      "--model=riaf", f"--output={output / 'responses'}"]))
    return steps, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="read-only dependency and GPU checks")
    doctor.add_argument("--backend", choices=("cpu", "openmp", "cuda"), default="cpu")
    doctor.add_argument("--build-dir", type=Path)
    doctor.add_argument("--json", action="store_true")
    inspect = sub.add_parser("inspect", help="check Stokes arrays, termination and physical flux scale")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--frame", type=int, default=0)
    inspect.add_argument("--freq-index", type=int, default=0)
    inspect.add_argument("--json", action="store_true")
    for command in ("build", "demo", "quickstart"):
        p = sub.add_parser(command, help={"build": "configure and build selected models", "demo": "run a data-free RIAF example with an existing executable", "quickstart": "build RIAF, run an example, inspect and plot the result"}[command])
        p.add_argument("--source-dir", type=Path, default=ROOT)
        p.add_argument("--build-dir", type=Path)
        p.add_argument("--backend", choices=("cpu", "openmp", "cuda"), default="cpu")
        p.add_argument("--json", action="store_true")
        p.add_argument("--dry-run", action="store_true", help="print the argument lists; create no files or builds")
        if command != "demo":
            p.add_argument("--models", default="riaf",
                           help="comma-separated models to build (default: riaf); readers and examples in docs/cli.md")
            p.add_argument("--coordinates", default="fmks",
                           help="GRMHD coordinate backends: fmks (default), mks, spherical_ks, cartesian_ks, boyer_lindquist, or all; separate several with semicolons")
            p.add_argument("--jobs", type=int, default=2, help="parallel compilation jobs (default: 2)")
            p.add_argument("--tests", action="store_true", help="also build and run the regression suite")
            p.add_argument("--trace", action="store_true")
            p.add_argument("--kokkos-dir", type=Path)
            p.add_argument("--nvcc-wrapper", type=Path)
            p.add_argument("--cuda-arch")
            p.add_argument("--cmake-arg", action="append", default=[], help="extra CMake setting, e.g. --cmake-arg=-DHDF5_ROOT=/path")
        if command != "build":
            p.add_argument("--example", choices=EXAMPLES, default="image")
            p.add_argument("--resolution", type=int, default=256,
                           help="image width and height (default 256; use 64 for a quick installation check)")
            p.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
            p.add_argument("--output-dir", type=Path)
            p.add_argument("--image-exe", type=Path)
            p.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = environment(args.backend, args.build_dir)
            emit(result, args.json); return 0 if result["ok"] else 1
        if args.command == "inspect":
            require_plot_dependencies(False)
            from kpolaris_image import image_summary, load_image
            result = image_summary(load_image(args.input, args.frame, args.freq_index))
            emit(result, args.json); return 0 if result["ok"] else 1
        root = args.source_dir.resolve()
        if not (root / "src/KPolaris.hpp").is_file() or not (root / "CMakeLists.txt").is_file():
            raise ValueError("source-dir is not a KPolaris source tree")
        build = args.build_dir.resolve() if args.build_dir else root / "build" / f"{args.backend}-quickstart"
        for field in ("jobs", "resolution", "threads"):
            if hasattr(args, field) and getattr(args, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if args.command == "quickstart" and (args.models != "riaf" or args.image_exe):
            raise ValueError("quickstart builds RIAF; use build for other models or demo --image-exe for an existing executable")
        steps = build_plan(args, root, build) if args.command != "demo" else []
        result = {"schema": "kpolaris.workflow.v1", "ok": False, "backend_requested": args.backend, "build_directory": str(build)}
        output, images = None, []
        if args.command != "build":
            default_output = ("quickstart" + ("-" + args.backend if args.backend != "cpu" else "")) if args.command == "quickstart" else args.example
            output = args.output_dir.resolve() if args.output_dir else root / "outputs" / default_output
            demo_steps, images = demo_plan(args, root, build, output)
            steps += demo_steps
            result["output_directory"] = str(output)
        if args.dry_run:
            result.update({"dry_run": True, "commands": [{"stage": n, "command": c} for n, c in steps]})
            emit(result, args.json); return 0
        if output:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError(f"output directory is not empty: {output}; choose a new --output-dir")
            require_plot_dependencies(not args.no_plot)
        if args.command != "demo" and not shutil.which("cmake"):
            raise ValueError("CMake is missing; run doctor and activate/install build dependencies")
        logs = output if output else build / "workflow"
        try:
            run_steps(steps, logs, root, result)
            if images:
                from kpolaris_image import image_summary, load_image
                result["images"] = []
                expected_backend = {"cpu": "Serial", "openmp": "OpenMP", "cuda": "Cuda"}[args.backend]
                for image in images:
                    log = (logs / (image.stem + ".log")).read_text()
                    actual = re.search(r"Default Device:\s*(\S+)", log)
                    result["execution_space"] = actual[1] if actual else None
                    if actual is None or actual[1].lower() != expected_backend.lower():
                        raise RuntimeError(f"requested {args.backend} but executable reported {result['execution_space']}; use the matching executable/build")
                    for fi in range(2 if args.example == "multifrequency" else 1):
                        check = image_summary(load_image(image, 0, fi))
                        result["images"].append(check)
                        if not check["ok"] or not (check["peak_I"] and check["peak_I"] > 0):
                            raise RuntimeError(f"demo did not produce a finite, nonempty image with all rays returned: {image}")
            result["ok"] = True
        except (RuntimeError, OSError, ValueError) as exc:
            result["error"] = str(exc)
        (logs / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        emit(result, args.json)
        return 0 if result["ok"] else 1
    except (OSError, ValueError) as exc:
        if getattr(args, "json", False):
            emit({"schema": "kpolaris.error.v1", "ok": False, "error": str(exc)}, True)
            return 2
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
