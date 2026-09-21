# Deployment

Use Python 3.11+, CMake 3.25+, a C++20 compiler, and HDF5 with C++ bindings.
Plotting/inspection requires the modules in `requirements.txt`. Respect an
existing conda, Spack, module or virtual environment; the helper does not install
system packages or edit shell configuration.

```bash
python3 scripts/kpolaris.py doctor --json
python3 scripts/kpolaris.py quickstart --backend=cpu --jobs=2 --output-dir=outputs/first-image --json
python3 scripts/kpolaris.py quickstart --backend=openmp --threads=4 --output-dir=outputs/openmp --json
```

The helper passes explicit backend flags and keeps separate build directories.
The backend commands on this page are alternatives: for initial deployment,
build only RIAF with the selected backend. For GRMHD, select only the reader and
coordinate backend required by the input. Avoid `--coordinates=all` unless the
workflow needs every coordinate backend.

Explain before starting that initial CUDA compilation of GRMHD readers can take
tens of minutes, depending on the host CPU, compiler and enabled features.
Compilation uses host CPU and RAM; an idle GPU or a quiet log alone does not mean
the build has stalled. Check compiler activity and available memory before
restarting. Lower image resolution and looser integration tolerances reduce
runtime, not compilation work. Keep the build directory and reuse executables
for subsequent images.

CMake downloads Kokkos by default. Offline/preinstalled dependency example:

```bash
python3 scripts/kpolaris.py quickstart --kokkos-dir=/path/to/lib/cmake/Kokkos \
  --cmake-arg=-DHDF5_ROOT=/path/to/hdf5 --output-dir=outputs/offline --json
```

CUDA requires the toolkit, working driver, and an architecture supported by
Kokkos and that toolkit. `doctor --backend=cuda --json` reports visible GPUs.
The helper infers architecture only when all detected GPUs yield one supported
architecture; otherwise provide `--cuda-arch`. `CUDA_VISIBLE_DEVICES` still controls
runtime device visibility. On heterogeneous hosts always choose the architecture
and device explicitly.

```bash
python3 scripts/kpolaris.py quickstart --backend=cuda --cuda-arch=AMPERE80 \
  --output-dir=outputs/cuda --dry-run --json
python3 scripts/kpolaris.py quickstart --backend=cuda --cuda-arch=AMPERE80 \
  --output-dir=outputs/cuda --json
```

The fetched-Kokkos path prepares `nvcc_wrapper` automatically in a bootstrap
build. External CUDA Kokkos needs both `--kokkos-dir` and `--nvcc-wrapper` from
its installation. A100 uses AMPERE80; architecture selection is not a performance
preset. Keep a fresh build directory when changing architecture or compiler.

Use `build --models=kharma --coordinates=fmks --build-dir=build/kharma` for a
KHARMA-only FMKS build. Add `--tests` only when regression checks are requested
or needed for a code change; use `--trace` when path output is required.
For single-frequency fast-light Stokes work, the optional CMake settings
`KPOLARIS_ENABLE_ANALYSIS_MODE=OFF`, `KPOLARIS_ENABLE_SLOW_LIGHT=OFF` and
`KPOLARIS_MAX_FREQUENCIES=1` omit unused kernels. Pass each through
`--cmake-arg=-DNAME=VALUE` and use a separate build directory. Do not disable
features needed by the requested workflow: diagnostic/response demos require
analysis support, and the multifrequency demo needs a multifrequency build.
New helper builds normally include those features and slow light.

Set `--jobs` according to host memory: the default is 2, and a CUDA compiler
process can consume several GiB. Use `--jobs=1` on memory-limited hosts; increase
concurrency only when enough RAM is available.
The normal CMake interface remains available for advanced/site-specific settings.
