# GRMHD inputs and image workflows

Use `quickstart` for a data-free RIAF example. To image your own GRMHD simulation,
build its reader, choose the matching coordinate backend, set the physical
parameters, and run the executable. The following commands assume the dependencies
in the [installation guide](quickstart.md) are available.

**Compile only what your data require.** Choose one reader and its coordinate
backend for the first run. Initial CUDA compilation can take tens of minutes;
retain the resulting executable for subsequent snapshots and camera settings.
The [selective-build guide](quickstart.md#selective-builds-and-existing-dependencies)
also shows how to omit slow light, physical diagnostics and shared multifrequency
support when your workflow does not need them.

## KHARMA: from a snapshot to Stokes images

For a KHARMA/Parthenon PHDF snapshot on an FMKS grid:

```bash
python3 scripts/kpolaris.py build --models=kharma --coordinates=fmks \
  --build-dir=build/kharma
cp params/kharma_recommended.par kharma.par
```

Edit `kharma.par` before running. In particular, set `kharma_mbh_solar`,
`kharma_M_unit`, `dsource`, the electron temperature prescription and the camera
for your simulation. The template's physical values are illustrative.

```bash
mkdir -p outputs/kharma
./build/kharma/kpolaris_model_image_kharma --parameter_file=kharma.par \
  --kharma_dump=/path/to/snapshot.phdf --coordinate=fmks --kharma_resample=none \
  --nx=256 --ny=256 --freq=230e9 --output=outputs/kharma/image.h5
python3 scripts/kpolaris.py inspect outputs/kharma/image.h5 --json
python3 scripts/plot_kpolaris_pol.py outputs/kharma/image.h5 \
  --layout=stokes --output=outputs/kharma/stokes.png
```

Replace `/path/to/snapshot.phdf` with your file. Command-line arguments override
the parameter file. For native MKS data, use `--coordinates=mks` at build time
and `--coordinate=mks` at runtime. Confirm the grid from the simulation metadata;
changing a filename extension or a coordinate option does not convert the input.

The explicit `kharma_resample=none` above samples the original grid. The shipped
template otherwise selects `spherical_ks_precomputed`, which requires a
`spherical_ks` build backend. Resampling introduces a second interpolation grid;
compare it with native sampling for your desired observables before using it.
The PHDF reader expects a complete, nonoverlapping meshblock decomposition of a
uniform global grid. Unsupported multilevel or incomplete layouts are rejected.

## iHARM/HARM

For an FMKS HDF5 dump, build and copy the matching template:

```bash
python3 scripts/kpolaris.py build --models=iharm --coordinates=fmks --build-dir=build/iharm
cp params/iharm_recommended.par iharm.par
```

Set the `iharm_*` physical parameters and camera in `iharm.par`, then run:

```bash
mkdir -p outputs/iharm
./build/iharm/kpolaris_model_image_iharm --parameter_file=iharm.par \
  --iharm_dump=/path/to/dump.h5 --coordinate=fmks --iharm_resample=none \
  --nx=256 --ny=256 --output=outputs/iharm/image.h5
python3 scripts/kpolaris.py inspect outputs/iharm/image.h5
python3 scripts/plot_kpolaris_pol.py outputs/iharm/image.h5 \
  --layout=stokes --output=outputs/iharm/stokes.png
```

Use MKS instead when appropriate to the dump. HDF5 inputs must carry the grid,
metric and primitive-variable metadata expected by the reader. The optional
`iharm_sks_primitives.par` and `iharm_sks_precomputed.par` templates use a
resampled spherical Kerr–Schild grid and require that compiled backend.

## AthenaK

AthenaK's meshblock binary input is sampled directly in Cartesian Kerr–Schild
coordinates. It requires the `cartesian_ks` backend:

```bash
python3 scripts/kpolaris.py build --models=athenak --coordinates=cartesian_ks \
  --build-dir=build/athenak
cp params/athenak_recommended.par athenak.par
```

Edit `athenak.par` for the physical normalization, electron prescription and
camera, then supply a compatible meshblock dump:

```bash
mkdir -p outputs/athenak
./build/athenak/kpolaris_model_image_athenak --parameter_file=athenak.par \
  --athenak_dump=/path/to/snapshot.bin --nx=256 --ny=256 \
  --output=outputs/athenak/image.h5
python3 scripts/kpolaris.py inspect outputs/athenak/image.h5
python3 scripts/plot_kpolaris_pol.py outputs/athenak/image.h5 \
  --layout=stokes --output=outputs/athenak/stokes.png
```

## BHAC

Build `--models=bhac --coordinates=mks --build-dir=build/bhac` for BHAC AMR
`.dat` input. Its executable is `build/bhac/kpolaris_model_image_bhac` and its
input key is `bhac_dump`. There is no bundled BHAC parameter template: use
`--help` to inspect the `bhac_*` parameters, then create a parameter file with
your grid metadata, units, electron prescription and camera. Run it with
`--parameter_file=bhac.par --bhac_dump=/path/to/snapshot.dat --output=bhac.h5`.

## Physical units and parameters

For all adjustable solver settings, see the [parameter reference](parameters.md).
The files in `params/` provide commented starting configurations; this section
highlights the physical inputs to check before using simulation data.

In the following table, replace `MODEL` with `iharm`, `kharma`, `athenak` or
`bhac` as appropriate.

| Parameter | Meaning |
| --- | --- |
| `MODEL_mbh_solar` | Black-hole mass in solar masses; sets the geometric length and time units |
| `MODEL_M_unit` | Simulation mass normalization in grams; sets the density and magnetic-field scales |
| `dsource` | Source distance in parsecs |
| `freq` | Observing frequency in Hz |
| `MODEL_trat_small`, `MODEL_trat_large`, `MODEL_beta_crit` | Electron-temperature prescription as a function of plasma beta |
| `MODEL_sigma_cut` | Magnetization threshold for excluding emission according to the model prescription |
| `inclination_deg`, `radius` | Viewing inclination in degrees and camera radius in geometric units |
| `fovx_dsource`, `fovy_dsource` | Angular field of view in microarcseconds |
| `outer_radius` | Outer limit of the transfer domain in geometric units |

Grid extent, radiating region and physical normalization must match the intended
model. Standard builds use thermal electrons; consult the
[electron-distribution guide](electron_distributions.md) before selecting other fits.
The physical magnetic-field polarity is retained. A field-reversal option
changes the source model and must not be used as a plotting convention.

## CPU threads and GPU builds

Add `--backend=cuda --cuda-arch=AMPERE80` to a build command for an A100 CUDA
build; select the architecture of your device and use a separate build directory.
The same input parameters are passed to the resulting executable.

For an OpenMP build, add `--backend=openmp`. Set the thread count when running
the solver, for example:

```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads \
  ./build/grmhd-openmp/kpolaris_model_image_kharma --kokkos-num-threads=8 \
  --parameter_file=kharma.par --kharma_dump=/path/to/snapshot.phdf \
  --coordinate=fmks --kharma_resample=none --output=kharma_openmp.h5
```

The [helper reference](cli.md) lists all build selections. `--jobs` is compile
parallelism; it does not set the solver's runtime threads.

## Frequencies, diagnostics and saved results

Set `freq_list=86e9,230e9,345e9` for multiple frequencies. The compiled
`KPOLARIS_MAX_FREQUENCIES` limits the number advanced together with shared
geometry; a larger list is processed in groups. Each frequency has its own
Stokes output and, when enabled, analysis arrays.

Add `analysis_mode=1` for physical diagnostics. See
[definitions and examples](diagnostics.md), including the distinction between
source contributions and parameter responses. Time-dependent input uses the
[slow-light workflow](slow_light.md) and requires a sequence covering every
radiation sample along the rays.

`parameter_output=auto` writes `<output>.params` containing the effective
configuration. Keep it with the HDF5 and input identifiers. Check ray termination
and finite Stokes before interpreting an image. The default `evpa_0=N` follows
image north toward east; the [conventions](polarization_conventions.md) define
the camera and polarization axes. Use the [HDF5 schema](hdf5_schema.md) for
array units and the [plotting guide](plotting.md) for selected frequencies.

## Ray traces and the binary model

`build --trace` also builds trace tools. Select a few representative rays before
storing large trajectory datasets. Trace sample spacing controls output density;
integration step controls govern numerical accuracy independently.

`binary_riaf` uses a superposed Kerr–Schild approximation. The supplied
`params/binary_riaf.par` works without an external trajectory. External trajectories
must satisfy the [trajectory schema](hdf5_schema.md#binary-trajectory-input-schema-kpolarisbinary_trajectoryv1).
`scripts/generate_binary_trajectory.py --help` describes an optional generator
using a user-supplied CBwaves source. Neither a long trajectory nor a successful
run extends the physical validity of the binary approximation.

## Repeated images and timing

`repeat_images=N` repeats the same fast-light calculation after loading the
model once. It writes numbered files such as `sequence_repeat0000.h5`; camera,
fluid and frequency settings remain fixed. It requires
`parameter_output=auto` or `none`. Existing numbered outputs are refused.
Use slow-light batches for different observer times.

`image_kernel_single` measures geometry and transfer integration;
`repeat_image_elapsed` includes the rest of each image operation after model
loading. Process wall time also includes startup and loading. State which
quantity you compare. `scripts/benchmark_image_scaling.py --help` describes
repeated resolution/platform measurements and host/GPU memory sampling; this
benchmark expects one frequency and one time per image.
