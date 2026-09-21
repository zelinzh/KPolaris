# First image and deployment

Run these commands from the source directory. KPolaris requires CMake 3.25+, a
C++20 compiler, HDF5 with C++ support, and Python 3.11+ for the supplied workflow
and plotting tools. Kokkos 5.1.1 is downloaded during configuration unless an
existing installation is selected. There are no required simulation downloads
for the RIAF examples. If you have not downloaded the source yet:

```bash
git clone https://github.com/zelinzh/KPolaris.git
cd KPolaris
```

Install or activate the build dependencies as described under
[Dependency environments](#dependency-environments) before continuing. Then
create a Python virtual environment and keep it active for subsequent commands:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/kpolaris.py doctor
python3 scripts/kpolaris.py quickstart
```

The first build compiles the RIAF image executable. The demo computes a 256 × 256
polarized image using the M87* example, checks that all rays returned and all
Stokes values are finite, and draws a total-intensity figure. Angular axes and a
linear brightness-temperature scale give a recognizable emission ring and
central dark region. A successful command exits with status 0. For a faster
installation check, add `--resolution=64`. Scientific work requires convergence
checks appropriate to the observable. See [the example model](first_image.md)
for the physical scales, interpretation and plotting commands.

The default output directory is `outputs/quickstart`. It contains the HDF5
image, its resolved parameter file, the PNG figure, command logs and `run.json`.
The JSON record includes image checks, the actual Kokkos execution space, and
the commands that ran. A mismatch between the requested and actual backend
fails the demo.
Existing nonempty output directories are refused: choose a new `--output-dir`
for another run. No system packages are installed by this workflow.

## Dependency environments

On Ubuntu 24.04, one installation route is:

```bash
sudo apt install build-essential cmake ninja-build libhdf5-dev python3-venv
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Activate the virtual environment again in a new shell. An existing HDF5/Python
module environment can be used instead; `doctor` identifies the interpreter and
packages in use.

If you already use Spack, `spack-cpu.yaml` provides an alternative environment:

```bash
spack env create kpolaris-cpu spack-cpu.yaml
spack env activate kpolaris-cpu
spack install
python3 scripts/kpolaris.py doctor
```

`spack-cuda.yaml` additionally selects a CUDA toolkit. Adjust that toolkit to
match your GPU, driver and host compiler before concretizing. The Python helper
does not require Spack and does not modify your dependency environment.

If Spack is already installed, you can also reuse its individual packages
without creating a new full environment. For example, after sourcing your
Spack `share/spack/setup-env.sh`:

```bash
spack install 'cmake@3.31:' 'hdf5+cxx~mpi'
spack load 'cmake@3.31:'
spack load 'hdf5+cxx~mpi'
```

Spack can install these dependencies in your user directory without sudo.
Then create or activate the Python environment and run the first-image commands
at the top of this page.
An HDF5 installation built with `~cxx` lacks the C++ interface and is not
sufficient, even if Python's `h5py` works. Activate the same dependency and
Python environments in each new terminal. `doctor` checks the active environment;
it does not search every installed package directory automatically.

For CUDA, also activate a compatible toolkit before running `doctor --backend=cuda`.
For example, `spack load cuda@12.9` selects an already installed CUDA 12.9 toolkit.
Having a working `nvidia-smi` only establishes driver/device availability;
compilation additionally requires `nvcc` on PATH.
CUDA executables retain the configured CUDA runtime and external library search
paths, including user-space installations. Reconfigure and rebuild if those
dependency installations move.

## Other backends

KPolaris uses Kokkos for portable parallel execution. Kokkos also provides
[HIP and SYCL execution spaces](https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html).
The helper recipes and platform validation in this release cover Serial,
OpenMP and CUDA; additional execution spaces require build integration and
validation on their target hardware.

CPU, OpenMP and CUDA use separate build directories. The CUDA toolkit, driver
and a suitable host compiler must already be available. CUDA architecture is
detected from the visible GPU; specify it explicitly for a cross build or a
machine with different GPU architectures.

```bash
python3 scripts/kpolaris.py quickstart --backend=openmp --threads=4 --output-dir=outputs/openmp
python3 scripts/kpolaris.py doctor --backend=cuda
python3 scripts/kpolaris.py quickstart --backend=cuda --output-dir=outputs/cuda
python3 scripts/kpolaris.py quickstart --backend=cuda --cuda-arch=AMPERE80 \
  --output-dir=outputs/a100-first-image
```

`--jobs=2` is the default compilation concurrency. Each CUDA compiler process
can use several GiB of host RAM. Use `--jobs=1` on memory-limited machines;
increase concurrency only when sufficient RAM is available. GPU architecture
names are Kokkos names (for example `AMPERE80`, `AMPERE86`, `ADA89`, `HOPPER90`,
`BLACKWELL120`). Changing architecture requires a new build directory. For direct CMake CUDA
presets, configure `cuda-bootstrap` first, then set the `KPOLARIS_CUDA_ARCH`
environment variable or pass `-DKPOLARIS_CUDA_ARCH=...` to CMake. The helper
performs these steps automatically.

## Selective builds and existing dependencies

**First CUDA builds can take tens of minutes**, particularly for GRMHD readers
with slow light, physical diagnostics and multiple shared frequencies enabled.
The host compiler generates and optimizes specialized GPU kernels; the GPU is
usually idle during this work. A quiet build log alone does not mean compilation
has stalled: check compiler activity and memory use before restarting a build.

Choose the smallest build that covers your workflow:

- For the first data-free image, use `quickstart`, which builds only RIAF.
- For simulation data, select the matching reader and coordinate backend.
  Avoid `--coordinates=all` unless you need all coordinate implementations.
- Add `--tests` for regression testing and `--trace` for trajectory output when
  those are part of your task. Ordinary image generation does not require them.
- Keep the build directory. Changing snapshots, camera settings, frequencies
  within the compiled capabilities, or resolution uses the existing executable.
  Changing build options, source or dependencies may trigger recompilation.

```bash
# Choose the reader matching your data; these are alternatives.
python3 scripts/kpolaris.py build --models=kharma --coordinates=fmks \
  --build-dir=build/kharma
python3 scripts/kpolaris.py build --models=athenak --coordinates=cartesian_ks \
  --build-dir=build/athenak
```

Reader executables are named `kpolaris_model_image_<model>`. The default compiled
GRMHD coordinate is FMKS; choose the backend matching your grid and resampling
settings. `--coordinates=all` includes the other GRMHD backends. The trace option
also builds the model trace tools. See the [helper reference](cli.md) for all
model choices and options, and the [GRMHD guide](user_guide.md) for complete
build, parameter-file, run, inspection and plotting examples.

### Optional single-frequency fast-light build

In a new build, the helper normally includes physical diagnostics, slow light
and capacity for two frequencies advanced together. If your task needs only single-frequency
fast-light Stokes images, omit the unused features at **build time**:

```bash
python3 scripts/kpolaris.py build --backend=cuda --models=kharma --coordinates=fmks \
  --build-dir=build/kharma-fast-cuda --jobs=2 \
  --cmake-arg=-DKPOLARIS_ENABLE_ANALYSIS_MODE=OFF \
  --cmake-arg=-DKPOLARIS_ENABLE_SLOW_LIGHT=OFF \
  --cmake-arg=-DKPOLARIS_MAX_FREQUENCIES=1
```

This retains full Stokes I/Q/U/V fast-light transfer. Slow light and optional
physical diagnostics/responses are unavailable, and shared multifrequency
transfer is not compiled. Enable the corresponding options in a separate build
when you need them. The diagnostic/response demos require analysis support.
The reduction in build time depends on the model and toolchain; no fixed
speedup is implied.

| Compile-time option | Keep it enabled when you need |
| --- | --- |
| `KPOLARIS_ENABLE_ANALYSIS_MODE` | Physical diagnostic maps or parameter responses |
| `KPOLARIS_ENABLE_SLOW_LIGHT` | Time-dependent transfer or geometric time-coverage probes |
| `KPOLARIS_MAX_FREQUENCIES` | Set to the number of frequencies to advance together; use 1 for single-frequency work |

Runtime settings such as `analysis_mode=0`, `slow_light=0`, `freq`, and image
resolution do not remove compiled kernels. Lowering the resolution or loosening
integration tolerances therefore does not shorten the build. After building,
use `demo` for RIAF examples or run the selected GRMHD executable directly.

### Existing dependencies and offline builds

An offline CPU build can use an existing compatible Kokkos installation:

```bash
python3 scripts/kpolaris.py build --kokkos-dir=/opt/kokkos/lib/cmake/Kokkos
```

An external CUDA Kokkos installation additionally needs
`--nvcc-wrapper=/opt/kokkos/bin/nvcc_wrapper`. Pass HDF5 or other CMake settings
as `--cmake-arg=-DHDF5_ROOT=/opt/hdf5`. The ordinary CMake interface remains
available; the Python helper is a convenience layer over it.

## Scripts and agents

Every workflow supports `--help`. Deployment plans can be inspected without
creating files or launching a build:

```bash
python3 scripts/kpolaris.py quickstart --dry-run --json
python3 scripts/kpolaris.py doctor --json
python3 scripts/kpolaris.py inspect outputs/quickstart/image.h5 --json
```

Exit status is 0 for success, 1 for failed execution or image checks, and 2 for
invalid input. `--json` keeps machine-readable output on stdout; progress and
log locations go to stderr. A failed build keeps its logs. An agent should read
`AGENTS.md` and `skills/kpolaris/SKILL.md`, inspect the environment, execute the
requested workflow, and report the actual validation outcome. The skill is
bundled with the repository and does not need an external service.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Python import fails | Use the same Python interpreter for pip and the workflow. Activate your virtual environment. |
| HDF5 is not found | Install its C++ development package or pass `-DHDF5_ROOT`. See `configure.log`. |
| Kokkos cannot be downloaded | Configure network access or supply `--kokkos-dir`. |
| CUDA compiler or architecture fails | Run `doctor --backend=cuda`; check toolkit/host-compiler compatibility and use an explicit architecture if needed. |
| Compiler is killed | Reduce `--jobs`; check available host memory. |
| Nonempty output directory | Select a new `--output-dir`; existing results are preserved. |
| A ray did not return or Stokes values are nonfinite | Inspect `run.json` and the image log. Check camera, integration domain, inputs and step limits before using the image. |
| Angular scale is unavailable | Plot in `M` or pixels, or provide the correct mass and distance. |

See [plotting](plotting.md), [the GRMHD guide](user_guide.md), and
[the HDF5 schema](hdf5_schema.md) for further use.
