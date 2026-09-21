# Command-line helper

`python3 scripts/kpolaris.py` checks dependencies, builds selected executables,
runs built-in RIAF examples and inspects saved images. Run it from the source
directory. It is a wrapper around CMake and the solver executables; physical
parameters for your own GRMHD calculations go to the solver.

## Subcommands

| Command | Purpose | Output |
| --- | --- | --- |
| `doctor` | Check the active compiler, Python packages, HDF5 and optional GPU environment | Dependency report; no installation or computation |
| `build` | Configure and compile selected models | Executables and `BUILD/workflow/run.json` with logs |
| `quickstart` | Build RIAF, run a built-in example, check it and plot it | HDF5, resolved parameters, figure, logs and `run.json` |
| `demo` | Run a built-in RIAF example using an existing build | The same result products without compilation |
| `inspect` | Check one saved frame and frequency | Stokes, ray termination and physical flux summary |

Use `--help` after any subcommand. `--json` writes a structured report to stdout;
progress goes to stderr. For `build`, `quickstart` and `demo`, `--dry-run --json`
prints the argument lists without creating files or running commands.

```bash
python3 scripts/kpolaris.py doctor --backend=cuda --json
python3 scripts/kpolaris.py build --models=kharma --build-dir=build/kharma --dry-run --json
python3 scripts/kpolaris.py inspect outputs/kharma/image.h5 --frame=0 --freq-index=0 --json
```

Exit codes are 0 for success, 1 for failed execution or image checks, and 2 for
invalid arguments. A dry-run report has `dry_run: true`; its `ok: false` means
the plan has not been executed. An executed example must report `ok: true`.

## Models and coordinates

`--models` is a **comma-separated build selection**, not a simulation filename.
The default is `riaf`. Each selected model produces
`kpolaris_model_image_<model>` in the build directory.

| Model | Input | Coordinate choice for a first build |
| --- | --- | --- |
| `riaf` | Analytic radiatively inefficient accretion flow; no external data | Built-in Boyer–Lindquist implementation |
| `torus` | Analytic magnetized torus; no external data | Built-in Cartesian Kerr–Schild implementation |
| `binary_riaf` | Approximate binary spacetime and analytic emitting flows | Built-in binary implementation; optional external trajectory |
| `iharm` | iHARM/HARM HDF5 snapshot | `fmks` or `mks`, matching the native grid |
| `kharma` | KHARMA/Parthenon PHDF snapshot | `fmks` or `mks`, with `kharma_resample=none` |
| `athenak` | AthenaK meshblock binary snapshot | `cartesian_ks` |
| `bhac` | BHAC AMR `.dat` snapshot | `mks` |

```bash
python3 scripts/kpolaris.py build --models=kharma --coordinates=fmks --build-dir=build/kharma
python3 scripts/kpolaris.py build --models=athenak --coordinates=cartesian_ks --build-dir=build/athenak
python3 scripts/kpolaris.py build --models=iharm,kharma --coordinates=fmks \
  --backend=openmp --build-dir=build/grmhd-openmp
```

`--coordinates` selects compiled GRMHD integration backends. Its default is
`fmks`; accepted names are `fmks`, `mks`, `spherical_ks`, `cartesian_ks`,
`boyer_lindquist` and `all`. Multiple names use a **quoted semicolon list**,
for example `--coordinates='fmks;cartesian_ks'`. `all` takes longer to compile.
This option does not convert the input grid or change its physical meaning.
The runtime `coordinate` and any resampling option must select a compiled
backend. RIAF and torus have their own fixed defaults and are not affected by
`--coordinates`.

For direct CMake, the model list also uses semicolons:
`-DKPOLARIS_IMAGE_MODELS='kharma;athenak'`. The helper uses commas for models;
it does not accept `--models=all`. Build only readers and coordinates required by
your workflow; the first CUDA build can take tens of minutes with GRMHD features
enabled. See [build scope and optional features](quickstart.md#selective-builds-and-existing-dependencies).

## Build options

| Option | Meaning |
| --- | --- |
| `--backend=cpu\|openmp\|cuda` | Serial CPU (default), threaded CPU, or CUDA build |
| `--build-dir=PATH` | Build location; defaults to `build/<backend>-quickstart` |
| `--source-dir=PATH` | Source checkout; defaults to the helper's own repository |
| `--jobs=N` | Compilation concurrency, default 2; distinct from runtime threads |
| `--tests` | Build **and run** the selected regression suite |
| `--trace` | Also build trace tools |
| `--kokkos-dir=PATH` | Use an existing Kokkos CMake package instead of downloading Kokkos |
| `--nvcc-wrapper=PATH` | Compiler wrapper for an existing CUDA Kokkos installation |
| `--cuda-arch=NAME` | Kokkos GPU architecture, e.g. `AMPERE80`; required if detection is ambiguous |
| `--cmake-arg=-DNAME=VALUE` | Extra CMake argument; repeat for multiple settings |

Keep separate directories for CPU, OpenMP, CUDA and different GPU architectures.
The helper enables analysis and compiles capacity for two shared frequencies;
slow-light support is also on by default in a new build.
Use `--cmake-arg=-DKPOLARIS_MAX_FREQUENCIES=4` to change that capacity; additional
frequencies can also run in groups. These are build choices, not imaging inputs.
For single-frequency fast light without physical diagnostics, see the
[optional smaller build](quickstart.md#optional-single-frequency-fast-light-build).
Runtime parameters and image resolution do not reduce compilation work.
See [installation](quickstart.md) for dependencies and offline configuration.

## Built-in examples

`quickstart` and `demo` run **RIAF only**. For GRMHD, use `build`, then run the
matching executable as shown in the [input guide](user_guide.md).

```bash
python3 scripts/kpolaris.py quickstart --backend=openmp --threads=8 \
  --output-dir=outputs/first-image
python3 scripts/kpolaris.py demo --backend=openmp --threads=8 \
  --example=multifrequency --output-dir=outputs/two-frequencies
```

| Example | Computation |
| --- | --- |
| `image` (default) | One polarized image at 230 GHz; plots total intensity |
| `multifrequency` | Images at 230 and 345 GHz |
| `diagnostics` | Radial source contributions and transfer diagnostics |
| `response` | Three runs: density, temperature and magnetic-field scale responses |

`--resolution=256` sets image width and height; 256 is the default.
`--resolution=64` is useful for a short installation check. `--threads=N`
sets runtime threads for an OpenMP example (default: up to four), whereas
`--jobs` controls compilation. `--no-plot` skips figures while still computing
and checking the image. `--image-exe=PATH` on `demo` selects an existing RIAF
executable; its execution backend must agree with `--backend`.

An existing nonempty output directory is refused. Choose a new `--output-dir`
for each run. The manifest records commands, logs, validation and the actual
execution space. The scripts do not silently substitute CPU for a requested GPU.

## Solver parameters

The compiled solver accepts `--parameter_file=FILE` and `--key=value` overrides:

```bash
./build/kharma/kpolaris_model_image_kharma --parameter_file=kharma.par \
  --kharma_dump=/path/to/snapshot.phdf --coordinate=fmks --kharma_resample=none \
  --freq=230e9 --nx=256 --ny=256 --output=outputs/kharma/image.h5
```

Set the input path and physical units in `kharma.par` first. Snapshot paths,
frequency, camera, mass normalization and integration controls belong here;
`kpolaris.py build` does not consume them. `parameter_output=auto` saves the
effective solver settings beside the output as `<output>.params`.

The [solver parameter reference](parameters.md) explains individual settings,
units, automatic values, aliases and interactions. The templates in `params/`
include grouped comments; they do not list every available option.
