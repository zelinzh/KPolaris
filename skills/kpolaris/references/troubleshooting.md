# Troubleshooting

Read `run.json` and the failing stage's log. The helper returns 1 for a failed
run/check and 2 for invalid input; `--json` keeps machine-readable output on stdout
and progress on stderr. A failed run never becomes success just because a plot exists.

- **Dependencies missing:** run `doctor` with the same Python and shell environment.
  HDF5 must have C++ bindings; set HDF5_ROOT/CMAKE_PREFIX_PATH if installed outside
  normal search paths. A missing `h5c++` hint alone does not prove HDF5 is absent.
- **Kokkos download unavailable:** use a local Kokkos installation via `--kokkos-dir`.
  Fetched and external CUDA Kokkos need compatible compilers and architecture.
- **CUDA architecture/toolkit mismatch:** inspect `doctor --backend=cuda --json`,
  choose the actual device, and use a new build directory. Do not reuse the
  bootstrap CPU build as the CUDA build.
- **Compile killed/OOM:** lower `--jobs`; build only the required model. CUDA
  compile time is separate from image generation time.
- **Output folder already exists:** choose a new `--output-dir`. Logs and images
  may be valuable; do not remove them to conceal a failed attempt.
- **Unknown frame/frequency:** `inspect` lists available groups. The plotting
  defaults select frame 0, frequency 0; pass explicit indices for other images.
- **Gray polarization pixels:** fractional/angle displays mask weak intensity,
  zero linear polarization and invalid rays. Inspect the mask and termination
  counts; this is not a claim that the local field or radiation vanishes.
- **Flux unavailable:** physical distance/length scale is missing. Do not guess
  an object distance or multiply by an invented factor. Use cgs intensity or M
  axes until the model supplies the physical normalization.
- **Invalid rays/time exhaustion:** inspect termination counts and the effective
  parameters. For slow light check time coverage; for integration failures check
  tolerance, minimum step and step budget before interpreting Stokes values.
