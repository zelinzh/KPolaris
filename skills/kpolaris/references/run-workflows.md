# Workflows

After quickstart, reuse its executable:

```bash
python3 scripts/kpolaris.py demo --example=multifrequency --output-dir=outputs/frequencies --json
python3 scripts/kpolaris.py demo --example=diagnostics --output-dir=outputs/diagnostics --json
python3 scripts/kpolaris.py demo --example=response --output-dir=outputs/responses --json
```

`--build-dir` and `--image-exe` select an existing build. `--no-plot` retains
HDF5 validation when Matplotlib is unavailable. All demos use the same M87*
analytic RIAF parameters (mass 6.5e9 solar masses, distance 16.8e6 pc). Images
and multifrequency runs default to 256² total-intensity figures; polarization
is available through the plotter. The plasma and spin are illustrative, not
an observational fit. Use `--resolution=64` only for an installation check.

Inspect and plot a selected frequency:

```bash
python3 scripts/kpolaris.py inspect outputs/frequencies/multifrequency.h5 --freq-index=1 --json
python3 scripts/plot_kpolaris_pol.py outputs/frequencies/multifrequency.h5 \
  --freq-index=1 --layout=stokes --output=outputs/frequencies/stokes.png
```

For real GRMHD input, build the matching reader, use its parameter template,
and supply simulation-specific inputs and units. Example command shape:

```bash
python3 scripts/kpolaris.py build --models=kharma --coordinates=fmks --build-dir=build/kharma --json
./build/kharma/kpolaris_model_image_kharma --parameter_file=params/kharma_recommended.par \
  --kharma_dump=/data/snapshot.phdf --kharma_resample=none \
  --nx=256 --ny=256 --output=outputs/kharma.h5 --parameter_output=auto
```

The path and template physical normalization are examples: resolve them from
the user's model. Do not interpret a successful input load as correct units.
Use `docs/parameters.md` for solver parameter meanings, units, defaults and
scope. Templates contain grouped comments; preserve the distinction between
physical source choices, numerical accuracy controls and cache settings.

For slow light supply ordered dump/time lists and the observer arrival time,
then run `--slow_light_time_probe=1` before a complete image. Use the reported
required fluid-time interval. Retain 0.1M data when that is the user's input;
resident windows and memory budgets change storage, not sampling cadence.
`slow_light_windows_per_block` and KHARMA `slow_light_snapshot_cache_gib` bound
residency; prefetch and pipeline flags overlap input with computation. Read the
public slow-light manual for batch-job syntax and DDC integration.
All supported GRMHD image backends default to `slow_light_step_mode=decoupled`
and `slow_light_interpolation=fluid`, including diagnostics, frequency groups and
observer-time batches. Data requests pause/retry an uncommitted step without
clipping to cache boundaries; no image workflow implicitly falls back to snapshot
stepping. The `continuous` input alias is saved as `decoupled`.
`block` and `snapshot` remain explicit legacy options. Frequency groups share
geometry and snapshots; `multifrequency_chunk_size=1` selects separate propagation
while still sharing input and Pass A. Capacity is bounded by `KPOLARIS_MAX_FREQUENCIES`.
Time probes are geometric and independent of the step policy. Trace interfaces
retain their single-frequency pairwise stepping.

Fast-light trace supports multiple frequencies; slow-light trace is single-frequency.
Start with a few rays. `trace_fields=all` stores geometry, wavevector, transported
basis and radiative quantities; request fewer fields for large traces. Trace output
sampling does not control the integrator's accuracy. `plot_trace_diagnostics.py`
handles compact ragged traces, and `bin_trace_physical_maps.py` bins sampled data.
