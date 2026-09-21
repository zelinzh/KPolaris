---
name: kpolaris
description: Build and use KPolaris for polarized black-hole images, GRMHD inputs, slow light, and ray diagnostics. Use for KPolaris installation, first-image checks, backend selection, reproducible runs, plotting, and troubleshooting.
---

# KPolaris

Locate the source tree containing `src/KPolaris.hpp`, `params/`, and
`scripts/kpolaris.py`. A copied skill does not include the solver itself; use
`--source-dir` or the user's checkout, not a path inferred from the installed skill.

## First successful image

From the source root:

```bash
python3 scripts/kpolaris.py doctor --json
python3 scripts/kpolaris.py quickstart --dry-run --json
python3 scripts/kpolaris.py quickstart --output-dir=outputs/first-image --json
```

`doctor` is read-only. The dry run lists literal argument arrays without downloading,
configuring or writing outputs. `quickstart` builds only the analytic RIAF image
target, runs it, checks finite Stokes and ray termination, and draws a figure.
Its default CPU backend is suitable when the user has not requested a backend.
Use the requested CPU/OpenMP/CUDA backend when one is specified; report a missing
dependency instead of silently substituting another device.

Read `run.json`: require `ok=true`, positive demo intensity, finite Stokes and
all rays returned. A zero process exit or an existing PNG alone is insufficient.
The report, effective `.params`, HDF5 and logs identify the actual run. An existing
nonempty output directory is refused; choose another output directory for a new run.

## Choose the workflow

- Installation or GPU configuration: [deployment](references/deploy.md).
- Multi-frequency, physical diagnostics, real GRMHD data, slow light, or plotting:
  [workflows](references/run-workflows.md).
- Failed commands, missing dependencies or invalid image pixels:
  [troubleshooting](references/troubleshooting.md).

Keep builds selective. `build --models=kharma --coordinates=fmks` builds that
reader; `--tests` adds the regression suite. RIAF examples need no external data.
For GRMHD, obtain the actual input path and physical normalization before running;
never invent simulation files, snapshot times, mass units or electron parameters.

## Scientific conventions

- `k` remains the future-directed physical photon wavevector. Pass A uses
  negative affine steps; Pass B uses positive steps with the transported screen.
  CLI `step`, `min_step`, `max_step` are positive magnitudes.
- All supported GRMHD image backends, spectra, diagnostics and observer-time
  batches default to `decoupled + fluid`.
  `continuous` is a compatibility alias for the stepping value, not a switch.
  Cache sizes affect residency/throughput, not integration partitioning. Four
  retained halo states are included in memory sizing. Frequency groups share
  geometry and input; each frequency has separate Stokes and diagnostic state.
  Image workflows do not implicitly fall back to snapshot stepping. Geometric
  time probes are independent of this policy; slow-light trace remains pairwise.
  Probe time coverage first; use explicit `block` or `snapshot` to replay legacy results.
- Physical source/response maps require `analysis_mode=1`. Source attribution
  retains foreground transfer; it is not a replacement simulation with that
  region removed. Use the documented parameter-response families for derivatives.
- Plot selected frame and frequency explicitly for libraries. Native Q/U/V use
  the recorded `evpa_0=N|W` output basis (default N); intermediate trace samples
  retain their transported basis. Image north does not fix a physical sky orientation. Flux requires a valid physical
  conversion scale. Do not normalize independent images before checking agreement.
  Do not mirror images, fit EVPA offsets, or flip Stokes to improve agreement.
  Read the coordinate/polarization conventions before any explicit basis conversion.
  Pinhole sampling defaults to pixel centers; the ipole -0.01-pixel offset is
  only an explicitly requested legacy sampling choice.
- The default demo computes a 256² M87* analytic RIAF image at 230 GHz and
  plots total intensity on a linear brightness-temperature scale. Mass and
  distance follow EHT 2019; the plasma and spin illustrate an accretion flow,
  not an observational fit. Use `--resolution=64` only for a fast installation
  check. Scientific use requires convergence checks for the observable.
