# Slow light, batches and optional DDC input

Slow light samples time-dependent fluid states along each ray. Supply a sequence
of snapshots with strictly increasing times covering every radiation sample.
A single snapshot is sufficient for fast light, but cannot represent this
time-dependent calculation.

## Time coverage and interpolation

With `slow_light=1`, camera-relative coordinate time starts at `x[0]=0` and

```text
t_fluid = slow_light_observation_time + x[0]
```

`slow_light_observation_time` is the coordinate arrival time at the camera.
Use `slow_light_time_probe=1` with your camera and radiation domain to determine
the required time support before choosing the sequence. Supply corresponding
`slow_light_dump_list` and `slow_light_time_list` values; their times must agree
with the snapshots. The observer time is not automatically the first dump time.

The default `slow_light_interpolation=fluid` interpolates spatially sampled
fluid quantities in time, reconstructs four-vectors from velocity and magnetic
primitives, then applies the magnetization cut and evaluates coefficients.
Derived scalar caches (`ne`, `Theta_e`, `B`, `sigma`, `beta`) retain the reader's
spatial reconstruction convention and are also interpolated; this does not
rederive every scalar from interpolated density and internal energy.

`slow_light_interpolation=coefficients` instead evaluates the two snapshots'
transfer coefficients and interpolates those coefficients in time. The two
orders differ at finite snapshot spacing. Refine the time spacing and compare
Stokes images, total flux and net polarization. Native-grid and direct
meshblock paths support fluid interpolation; a pre-resampled input retaining
only four-vector fluid states does not. For native KHARMA, set
`kharma_resample=none` explicitly.

Fast/slow comparisons also need a defined reference event. For example,
`t_ref=t_obs-camera_radius` removes a common distant-camera delay, but is not an
exact Kerr light-travel-time correction. Record the chosen alignment.

## Integration and memory controls

The default image workflow uses `slow_light_step_mode=decoupled`: cache
boundaries do not truncate integration steps. Missing time samples pause a ray
until the required states are resident; no partial step is accepted. This
applies to the iHARM, KHARMA, AthenaK and BHAC image paths, including multiple
frequencies, diagnostics and observer-time batches.

```ini
slow_light=1
slow_light_windows_per_block=16
slow_light_prefetch=1
slow_light_prefetch_snapshots=32
```

| Setting | Purpose |
| --- | --- |
| `slow_light_windows_per_block=N` | Maximum new time windows in a resident block; default 1 |
| `slow_light_prefetch_snapshots=H` | Host prefetch queue; 0 selects N automatically, with one additional read in progress |
| `slow_light_pipeline=1` | Overlap KHARMA upload and integration |
| `slow_light_snapshot_cache_gib` | KHARMA device snapshot-array budget; 0 disables automatic sizing |

Larger blocks reduce scheduling overhead at the cost of device memory.
For example, `slow_light_snapshot_cache_gib=15` caps the snapshot allocation
estimate at 15 GiB; it is **not** a limit on total process or GPU memory.
If N and a budget are both specified, the smaller permitted block is used.
Other readers use N directly; the GiB estimator currently applies to KHARMA.

Let D be the materialized device bytes per snapshot and K=4 the retained
history states in decoupled mode. A current block uses at most `(N+1+K)D`;
the pipeline pool uses `(2N+1+K)D`. Automatic sizing additionally reserves the
initial model: without the pipeline, `N=floor(budget/D)-2-K`; with it,
`N=floor((floor(budget/D)-3-K)/2)`. At least seven/nine snapshot payloads are
needed, respectively. Reserve extra memory for the runtime, ray states,
frequencies and diagnostic arrays. Compressed file size is not D. The prefetch
queue uses host RAM and does not control an external decoder's cache.

Legacy `block` and `snapshot` modes truncate steps at block or snapshot
boundaries; they remain available for reproducing earlier results.
`continuous` is an alias of `decoupled`. Geometric time probes use no snapshot
boundary stepping; the separate trace tool still advances through snapshot
pairs. Output records the actual policy. Cache independence does not imply
converged transfer: check geometric and radiation controls independently.

## Observer-time batches

`slow_light_batch_jobs` reuses camera geometry and resident fluid windows for
several observation times. Each time has independent Stokes and diagnostic
state, and writes one HDF5 file. It supports the readers above; KHARMA accepts
PHDF or optional native DDC input. All jobs share the camera, metric, physical
units, electron model, frequencies and integration/diagnostic settings.

Each job line contains observer time, first snapshot index, last snapshot index
and output path. Indices are zero-based and inclusive:

```text
# observer_time first_index last_index output
1200.0 0 100 /output/frame_000.h5
1200.5 5 105 /output/frame_001.h5
```

These numbers illustrate the format only. Use the time probe and actual
snapshot table to determine your ranges. Trim the global list to the union of
the jobs' support, starting its earliest entry at index zero. Paths may be
quoted; duplicate or existing output files are rejected.

```bash
./build/kharma/kpolaris_model_image_kharma \
  --parameter_file=physics.par --parameter_file=time_support.par \
  --slow_light=1 --slow_light_batch_jobs=jobs.txt \
  --slow_light_windows_per_block=16 --slow_light_prefetch_snapshots=32
```

Save the jobs file and global time table with each output's effective parameter
file. Batch outputs record their index and residency settings. Legacy block-mode
replay retains the shared prefix needed to reproduce its original block alignment.
Run time probes separately from image batches.

## Multiple frequencies

`freq_list=230e9,345e9,86e9` selects frequencies. The CMake capacity
`KPOLARIS_MAX_FREQUENCIES` limits the number advanced in one shared-geometry
kernel (raw CMake default 1; the helper builds capacity 2). With capacity 2,
`multifrequency_chunk_size=0` or 2 shares geometry for two frequencies; use 1
for sequential frequency groups. All groups reuse input and Pass A geometry.
Each frequency retains independent transfer and diagnostics.

The shared step is constrained by every frequency in a group, so changing the
group size can alter finite-step numerical results. Check convergence when
comparing groupings. Output analysis is under `/frame_0/freq_i/analysis` for
multiple frequencies and `/frame_0/analysis` for a single frequency.

## DDC input

DDC compression and decoding are maintained separately. KPolaris provides the
KHARMA client, device residency and GRRT calculation. Start a compatible native
DDC service, then set:

```ini
kharma_ddc_native=1
kharma_ddc_socket=/tmp/kpolaris-ddc.sock
kharma_ddc_manifest=/data/sequence_manifest.json
kharma_ddc_timeout_seconds=7200
kharma_dump=ddc_frame_00001.phdf
kharma_resample=none
```

The number in a native frame name is its manifest sequence index, not its
physical time. Ordinary paths still use the PHDF reader. Invalid protocols,
inconsistent times or nonfinite input are rejected. The timeout applies to each
blocking communication, not to the entire batch.

The compatibility service entry point needs a separately installed DDC project:

```bash
python3 scripts/serve_ddc_parallel.py --codec-project=/path/to/ddc-project \
  --manifest=/data/sequence_manifest.json --socket=/tmp/kpolaris-ddc.sock \
  --cache-dir=/local/ddc-cache
```

It does not include the compressor or replace the external compact service.

## Optional compact CUDA transport

This optional path is disabled by default. It requires CUDA Kokkos and matching
DDC headers supplied through CMake:

```text
-DKPOLARIS_ENABLE_DDC_COMPACT_CUDA=ON
-DKPOLARIS_DDC_CODEC_INCLUDE_DIR=/path/to/ddc/src/dense_dump_codec/include
```

The directory must provide `ddc_gpu_reconstruct.cuh`, `ddc_compact_receive.hpp`
and `ddc_compact_types.hpp`. Record the exact DDC version. On the service, select
native compact transport and its compatible spatial working cache, for example
`--transport=native --native-transfer-mode=compact --native-radius-max=auto`
and `--native-working-cache=/local/working-cache`. In the client environment,
set `KPOLARIS_DDC_COMPACT=1`. This requires retained anchors, a valid working
cache, and native MKS/FMKS input with `kharma_resample=none`.

CPU decompression supplies residuals, scales, exceptions and anchors; GPU
reconstruction produces float32 primitives for normal unit conversion and
derived quantities. It does not decompress bzip2/LZ4 on the GPU. For ordinary
pinned float32 reception only, set `KPOLARIS_DDC_PINNED=1` without COMPACT.
Only the literal value `1` enables these environment switches. Save them and
the service configuration alongside the solver parameters. Downstream builds
using compact headers must also provide the compatible DDC include directory.

## Accuracy and timing

Compare DDC/PHDF images of the same decoded states to check the interface;
compare decoded states with original fluid data to measure compression error.
CPU/GPU reconstruction roundoff is a third, separate comparison.

Report preparation, input waiting, upload/materialization, geometry and transfer
times with their overlap. Decoder time may already be included in client
waiting and must not be added again. Batch wall time divided by image count
measures throughput, whereas a single request's elapsed time measures latency.
Record the image settings, snapshot spacing, memory controls and accuracy with
each timing result.
