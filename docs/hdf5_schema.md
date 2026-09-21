# KPolaris HDF5 schemas and compatibility policy

This document is the data dictionary for HDF5 products written by the formal
KPolaris image and trace executables. It describes the current on-disk
contract, not the internal C++ layout. The image schema is
`kpolaris_image` version 1. Single-frequency trace files use
`kpolaris_trace` version 2; shared-geometry multi-frequency trace files use
`kpolaris_trace` version 3. The optional analysis subgroup has its own
`kpolaris_physical_analysis` version 2 marker. Readers may still encounter the
earlier trace v1/v2 and physical-analysis v1 layouts described below as
legacy development products.

The paper release must archive this document with the exact source tag and
validation data. Development files with a dirty or unknown source revision are
useful for testing but are not final paper evidence.

## Reader rules

A reader must first check the root `schema` string and integer
`schema_version`. It must reject a different schema name. It should reject a
newer major on-disk version unless it explicitly implements that version.
Within a recognized version, readers must ignore unknown attributes, groups,
and datasets so that optional additive metadata remains compatible.

Image readers must also check the per-pixel termination code. KPolaris writes
zero Stokes values for rays that did not reach the camera; zero is therefore
not by itself proof of a valid dark pixel. A science-valid image mask is

```python
valid = h5["/frame_0/diagnostics/reason"][...] == 1
```

The integer termination codes are:

| Code | Name | Meaning |
|---:|---|---|
| 0 | `none` | no terminal classification was assigned |
| 1 | `reached_camera` | Pass B returned to the camera surface |
| 2 | `reached_inner_boundary` | the ray reached the configured inner boundary |
| 3 | `escaped_domain` | the ray left the configured domain |
| 4 | `max_steps` | the integration step limit was reached |
| 5 | `invalid_state` | the integrator encountered a non-physical or non-finite state |
| 6 | `slow_light_time_exhausted` | the requested slow-light time lies outside the supplied dump windows |
| 7 | `adaptive_step_underflow` | adaptive error control or floating-point resolution prevented a valid step; excludes a rounding-sized remainder at a snapshot boundary |
| 8 | `metric_time_exhausted` | the dynamic metric trajectory does not cover the requested ray event time |
| 9 | `reached_direct_turn` | Pass A reached the selected direct-emission boundary; a successful final image still reports code 1 |

Only code 1 is currently included in image Stokes sums and fluxes. Readers
should treat all other values, including unknown future values, as invalid
unless the relevant schema revision says otherwise.

## Common provenance

Every image and trace file has the following root attributes. Image files
duplicate the same fields in `/header`.

| Attribute | Type | Meaning |
|---|---|---|
| `schema`, `schema_version` | string, int32 | schema identity and on-disk version |
| `code`, `code_version` | string | `KPolaris` and its configured semantic version |
| `code_revision` | string | source revision captured at CMake configure time |
| `code_source_dirty` | int32 | `0` clean, `1` dirty, `-1` unknown |
| `code_source_fingerprint` | string | SHA-256 of revision, tracked build-input diff, and untracked build-input contents; `unknown` only when unavailable |
| `compiler_id`, `compiler_version` | string | compiler provenance |
| `build_type` | string | configured CMake build type |
| `model`, `camera`, `coordinate` | string | selected physical model, camera, and coordinates |
| `nx`, `ny`, `nfreq` | int32 | image grid and frequency counts |
| `frequency_hz`, `frequency_list_hz` | float64, string | first frequency and complete serialized list |

Input dump paths and parameter-file paths are recorded when applicable. Paths
identify the local input but do not replace an archived file checksum. Final
paper manifests must therefore record SHA-256 values for source dumps,
parameter files, and HDF5 products outside this schema.

The format intentionally has no build timestamp: identical source and CMake
inputs should not acquire a different identity merely because they were built
later.

## Image schema: `kpolaris_image` v1

All image arrays use row-major logical axes `[iy, ix]` and HDF5 shape
`(ny, nx)`. `/grid/x[ix]` and `/grid/y[iy]` are pixel-center coordinates; the
`/grid` attribute `units` specifies their unit. `/grid/frequency_hz[fi]` maps
frequency index `fi` to observing frequency.

### Required structure

| Path | Contents |
|---|---|
| `/header` | structured copy of schema, provenance, layout, and Stokes-convention attributes |
| `/grid` | one-dimensional `x`, `y`, and `frequency_hz` float64 datasets |
| `/parameters/camera` | effective grid, field-of-view, distance, and flux-conversion attributes |
| `/parameters/spacetime` | metric, coordinates, mass scale, and spin attributes |
| `/parameters/integration` | effective step controls, boundaries, and optional-mode attributes |
| `/parameters/radiation` | model, frequencies, source inputs, and model-specific attributes |
| `/results` | scalar summary attributes and frequency-indexed float64 datasets |
| `/frame_0/freq_<fi>` | Stokes arrays and frequency-specific attributes |
| `/frame_0/diagnostics` | per-pixel integration and transported-basis diagnostics |

`/frame_0` is the only frame currently written, but its explicit index keeps
the layout extensible. Consumers must not assume that `freq_0` is 230 GHz; they
must read `frequency_hz`.

Shared GRMHD image implementations write common resolved quantities with a
`grmhd_*` prefix and loader-specific aliases with the active model prefix.
For `model="iharm"`, `kharma_*` attributes must be absent; for
`model="kharma"`, `iharm_*` attributes must be absent. This invariant applies
to root/frequency metadata and `/parameters/radiation`, and prevents products
from being attributed to the wrong input loader.

BHAC output uses the `bhac_*` metadata prefix and
`bhac_backend="direct_bhac_mks_amr"` to identify its AMR input format.

For `model="binary_riaf"`, the radiation metadata include
`binary_sampled_min_inverse_denominator` and
`binary_sampled_min_fluid_slice_timelike_margin`.  The first is the minimum
sampled Woodbury denominator used by the Superposed Kerr--Schild signature
audit.  The second is the minimum sampled value of
`-q_mu g^{mu nu} q_nu`, where `q=dT_A` is each hole's instantaneous rest-time
covector, over the emitting mini-disk domain.  Both are host-side
admissibility diagnostics evaluated before a render; a non-positive fluid
slice margin is rejected rather than converted into zero emissivity.

### Stokes images and fluxes

For every `fi`, `/frame_0/freq_<fi>` contains float64 datasets:

| Dataset | Meaning | Unit |
|---|---|---|
| `I`, `Q`, `U`, `V` | observer-frame Stokes specific intensity, `S_nu = nu^3 S_inv` | cgs specific intensity (`erg s^-1 cm^-2 Hz^-1 sr^-1`) |
| `I_inv`, `Q_inv`, `U_inv`, `V_inv` | invariant Stokes variables used by transport, `S_nu/nu^3` | invariant transport units |

The frequency group records
`stokes_convention="camera_frame"`,
`polarization_basis="camera_screen"`, and the selected `evpa_0="N"` (default)
or `"W"`. Image north is up and east is left by convention. N applies the
passive basis transformation `(I,Q,U,V)=(I_camera,-Q_camera,-U_camera,V_camera)`;
W preserves the internal camera values. Image coordinates and physical
polarization directions do not change. All observed Stokes products, including
fluxes, radial/source contributions and response derivatives, use that choice.
A celestial position angle remains an external orientation of the model.

Older files labelled `evpa_0="camera"` (or without the attribute) use W.
To reproduce their raw Q/U values, set `evpa_0=W` explicitly when rerunning an
old parameter file. Readers must not blindly flip Q/U in new N files.

The root image/trace attributes also declare `polarization_conventions_version=2`,
`stokes_definition`, `electric_field_phase_convention`, `screen_handedness`,
`evpa_definition`, `camera_azimuth_convention`, `sky_orientation`,
`image_pixel_order`, `image_axis_geometry`, and `output_stokes_transform`.
Images record `image_array_order="y,x"` and `image_display_origin="lower"`;
Trace samples, coefficients, `e1/e2` and `final_propagated_*` remain in the
internal transported screen; `final_observed_*` and the derived
`final_evpa_wrapped_rad` use the selected N/W output basis. Per-sample EVPA
histories are unchanged. Camera-basis overlap diagnostics also remain internal.
See [the full conventions](polarization_conventions.md).
N/W changes the Q/U reference axes, not array units or ordering.

`intensity_to_flux_jy_per_pixel` converts a sum of physical Stokes intensity
pixels to Jy:

```text
flux_S_Jy = sum(S_pixels) * intensity_to_flux_jy_per_pixel
```

The scale is valid only when `flux_scale_valid == 1`. When the physical length
or source-distance scale is unavailable, the scale and derived flux values are
NaN. It is repeated at the root, under `/parameters/camera`, `/results`, and
each frequency group. `/results/flux_{I,Q,U,V}_Jy_by_freq[fi]` is the canonical
multi-frequency summary; the scalar `flux_{I,Q,U,V}_Jy` attributes are aliases
for frequency index 0.

`/results/sum_{I,Q,U,V}_inv_by_freq` contains the unweighted pixel sums of the
invariant arrays. It must not be interpreted as a cgs intensity or Jy value.

### Per-pixel diagnostics

Every diagnostic has shape `(ny, nx)`. Integer datasets are int32 and all
others are float64.

| Dataset | Meaning |
|---|---|
| `reason` | termination code from the table above |
| `pass_a_steps` | accepted camera-to-source geometric steps |
| `steps` | accepted source-to-camera Pass B steps |
| `total_steps` | `pass_a_steps + steps` |
| `closure_x` | spatial camera-return position mismatch |
| `closure_k` | maximum component mismatch of the returned wave vector |
| `final_null` | final metric contraction `g(k,k)` |
| `frame_error` | maximum transported-frame orthonormality error |
| `det_r` | determinant of the 2-D final camera-basis overlap matrix |
| `overlap_r11`, `overlap_r12`, `overlap_r21`, `overlap_r22` | final-to-initial camera-screen basis overlap matrix |
| `basis_identity_error` | maximum absolute difference of the overlap Gram matrix from identity |
| `basis_rotation_angle` | rotation inferred from the final screen-basis overlap, in radians |

`/results` also stores maxima, warning counts, step summaries, and the pixel
indices at which selected maxima occurred. These are derived summaries; the
per-pixel datasets remain authoritative for a new analysis.

### Optional physical-analysis subgroup

When `analysis_mode=1`, the analysis group has
`schema="kpolaris_physical_analysis"` and `schema_version=2`. A
single-frequency image uses `/frame_0/analysis`; a multi-frequency image uses
`/frame_0/freq_<fi>/analysis` so every frequency has an independent physical
analysis. The `/header` attribute `analysis_layout` records the active layout.
All image maps have shape `(ny, nx)`; `dominant_emission_region` and
`radiation_substeps` are int32, and the other maps are float64.

The emission weight is `w = max(j_I, 0) |d lambda|`. A prefix
`emission_weighted_` denotes `integral(w q) / integral(w)`. A prefix
`dominant_` denotes the single sample with the largest `w`, not a mean.

| Dataset | Definition |
|---|---|
| `radiating_path_length` | `integral |d lambda|` over radiating samples |
| `emission_weight` | `integral max(j_I,0) |d lambda|` |
| `emission_weighted_radius` | emission-weighted Boyer--Lindquist-like radius in `M` |
| `emission_weighted_optical_depth_to_camera` | emission-weighted remaining scalar `alpha_I` depth to the camera |
| `absorption_depth` | `integral max(alpha_I,0) |d lambda|` |
| `absorption_operator_depth` | path integral of the polarized absorption-operator norm |
| `faraday_rotation_depth` | signed `integral rho_V |d lambda|` in the propagated screen basis |
| `faraday_conversion_depth` | `integral sqrt(rho_Q^2+rho_U^2) |d lambda|` |
| `faraday_operator_depth` | path integral of the full Faraday-operator norm |
| `dominant_emission_radius` | radius of the largest-weight sample, in `M` |
| `dominant_emission_region` | 0 no positive weight; 1 `r<5M`; 2 `5M<=r<20M`; 3 `r>=20M` |
| `dominant_ne_cgs`, `dominant_thetae`, `dominant_b_cgs`, `dominant_beta`, `dominant_sigma` | plasma state at the largest-weight sample |
| `emission_weighted_ne_cgs`, `emission_weighted_thetae`, `emission_weighted_b_cgs`, `emission_weighted_beta`, `emission_weighted_sigma` | emission-weighted plasma state |
| `photon_ring_winding_estimate` | accumulated `abs(delta phi)/(2 pi)` |
| `radiation_substeps` | accepted samples that participated in radiation transport |

`ne_cgs` is in `cm^-3`, `b_cgs` is in gauss, and `thetae`, plasma beta, sigma,
and optical/Faraday depths are dimensionless. Affine path length and emission
weight retain the active model's transport normalization and should be used as
relative diagnostics unless the archived model metadata supplies the complete
physical conversion.

The largest-weight (`dominant_*`) sample depends on the adaptive step partition.
`photon_ring_winding_estimate` is a coordinate-azimuth proxy, not a photon-ring
order. In the analytic RIAF diagnostics, beta and sigma are currently zero
placeholders, not inferred plasma values. None of the camera-return basis
errors should be interpreted as an observable gravitational polarization angle.

Analysis v2 adds an observer-weighted radial decomposition. The group
attributes `radial_bins`, `radial_min`, `radial_max`, and
`formation_fraction` specify the logarithmic grid and central formation
fraction. `radial_bin_edges` and `radial_bin_centers` have shapes `(Nr+1,)`
and `(Nr,)`. Every `radial_*` image cube uses bin-major shape
`(Nr, ny, nx)`.

For each bin, only emission generated inside that bin is added to its Stokes
subvector; the complete absorption/Faraday operator propagates every subvector
to the observer. Thus the four `radial_stokes_*_inv_contribution` cubes are in
the observer basis and must sum over axis 0 to the ordinary `*_inv` image.

| Dataset | Definition |
|---|---|
| `radial_stokes_{I,Q,U,V}_inv_contribution` | observer-basis invariant Stokes contribution from source terms emitted in each radial bin |
| `radial_absorption_depth` | raw `max(alpha_I,0)|d lambda|` accumulated by radial bin |
| `radial_faraday_rotation_depth` | signed `rho_V|d lambda|` accumulated by radial bin |
| `radial_faraday_conversion_depth` | `sqrt(rho_Q^2+rho_U^2)|d lambda|` accumulated by radial bin |
| `radial_faraday_operator_depth` | Faraday operator-norm depth accumulated by radial bin |
| `observer_weighted_radius_I` | radius weighted by `max(I_k,0)` |
| `observer_weighted_radius_linear` | radius weighted by `sqrt(Q_k^2+U_k^2)` |
| `observer_weighted_radius_circular` | radius weighted by `abs(V_k)` |
| `{intensity,linear,circular}_formation_radius_{low,median,high}` | radial quantiles for the corresponding non-negative contribution weight |
| `los_linear_coherence` | `abs(sum_k(Q_k+iU_k))/sum_k abs(Q_k+iU_k)` |
| `los_circular_coherence` | `abs(sum_k V_k)/sum_k abs(V_k)` |
| `contribution_closure_max_abs` | maximum absolute component residual between the bin sum and ordinary observed Stokes |
| `contribution_closure_relative_l1` | four-component residual L1 divided by the radial-contribution L1 |
| `foreground_*` | depth in radial bins whose centers exceed `intensity_formation_radius_high` |
| `foreground_faraday_operator_fraction` | foreground Faraday operator depth divided by the total |

The source bins label **seed emission**, not where propagation later creates
or converts a Stokes component. In particular a V-weighted source radius need
not locate Faraday conversion. Bin-local cancellation has already occurred
inside each contribution, so LOS coherences depend on bin width. The first
and last bins also collect emission below/above the configured radial bounds.
The additive budget and fixed-operator response are described in
[the diagnostics guide](diagnostics.md) and
implemented by `scripts/analyze_polarization_budget.py`. Its global LOS
coherence uses the source cubes directly, including exactly cancelling pixels.

The `foreground_*` fields are explicitly an outer-radius proxy, not a
path-ordered foreground assertion: strongly lensed rays need not be monotonic
in radius. Use trace v2/v3 `post_freeze_*` products for a path-ordered test.
`observer_weighted_available=1` indicates the complete v2 decomposition. A
legacy slow-light v2 development product may set it to zero and retain only
the v1 line-of-sight maps; paper products must require a value of one.

## Trace schema: `kpolaris_trace` v2 and v3

Trace output is deliberately field-selectable. The root attributes
`trace_fields`, `trace_precision`, `trace_layout`, and `trace_compression`
record the request. A dataset omitted by `trace_fields` is not an error.
Every trace also contains `/parameters/camera`, `/parameters/spacetime`,
`/parameters/integration`, and `/parameters/radiation`. These groups record the
effective camera and integration controls, Kerr coordinate/spin, observing
frequencies, active model inputs, effective emission type/fit, and nonthermal
parameters. The root and radiation group record the active dump path when
applicable, and `/parameters` records the parameter-file path when supplied.
Unless `parameter_output=none`, trace writes a replayable effective parameter
file to `<output>.params` (or the explicitly requested path) after resolving
the active model and defaults. The HDF5 root and `/parameters` group record
that sidecar path as `effective_parameter_file`. The sidecar contains only
parameters accepted by the trace parser; derived FMKS quantities are retained
as comments rather than non-replayable keys.
For a full-step analysis oracle, readers must require these physical attributes
to match the image before comparing reconstructed maps.

The required `/rays` datasets are:

| Dataset | Type and shape | Meaning |
|---|---|---|
| `pixel`, `ix`, `iy` | int32, `(nray,)` | source image pixel identity |
| `sample_count` | int32, `(nray,)` | recorded samples per ray |
| `sample_offset` | int64, `(nray+1,)`, ragged only | cumulative offsets into flat trace arrays |
| `pass_a_steps`, `pass_b_steps`, `reason` | int32, `(nray,)` | integration counts and termination code |
| `closure_x`, `closure_k`, `final_null`, `frame_error` | selected trace precision, `(nray,)` | final ray diagnostics |
| `final_propagated_{I,Q,U,V}_inv` | selected trace precision, `(nray,)` | final invariant Stokes in the transported screen basis |
| `final_observed_{I,Q,U,V}_inv` | selected trace precision, `(nray,)` | final invariant Stokes in the selected N/W output basis |

The default `trace_layout="ragged"` stores every selected field as one flat
array of length `sample_offset[-1]`. Samples for ray `r` occupy
`field[sample_offset[r]:sample_offset[r+1]]`, and the slice length must equal
`sample_count[r]`. The legacy-compatible `dense` layout instead uses shape
`(nray, max_trace_samples)` and omits `sample_offset`; only the first
`sample_count[r]` entries of each row are valid.

Selected floating datasets use float32 for `trace_precision="float"` and
float64 for `trace_precision="double"`. This controls storage only; internal
scientific integration remains in the configured main precision. When
`trace_compression` is 1--9, trace arrays and ragged count/offset arrays use
chunked gzip/DEFLATE at that level. Image v1 arrays are currently contiguous
and uncompressed.

### Field sets

| Selector | Datasets |
|---|---|
| `lambda` | `lambda`, `dlambda` |
| `coords` | `r`, `theta`, `phi` |
| `plasma` | `ne_cgs`, `thetae`, `b_cgs`, `beta`, `sigma`, `nu_fluid_hz`, `theta_bk`, `b1`, `b2`, `cos2chi`, `sin2chi` |
| `x` | `x0`, `x1`, `x2`, `x3` |
| `k` | `k0`, `k1`, `k2`, `k3` |
| `e1` | `e10`, `e11`, `e12`, `e13` |
| `e2` | `e20`, `e21`, `e22`, `e23` |
| `coeffs` | `jI`, `jQ`, `jU`, `jV`, `aI`, `aQ`, `aU`, `aV`, `rhoQ`, `rhoU`, `rhoV` |
| `stokes` | invariant accumulated `SI`, `SQ`, `SU`, `SV` at recorded samples |

`minimal` and `default` select `lambda,coords,coeffs,stokes`; `state` selects
`x,k,e1,e2`; `all` selects every set. `r`, `theta`, and `phi` are the
Boyer--Lindquist-like physical coordinates returned by the active radiation
model even when the integration coordinates differ. Coefficients and Stokes
values use the same invariant transport normalization as the solver.

### Derived polarization histories

When the selected fields include `lambda`, `coords`, `coeffs`, and `stokes`, a
single-frequency trace contains `/derived`; each multi-frequency
`/trace/freq_<fi>` contains its own `derived` subgroup. Its `available`
attribute is one. If required source fields were omitted, the subgroup remains
present with `available=0` and an `unavailable_reason`.

Per-ray datasets have shape `(nray,)`. `complete=1` requires unit stride and
`sample_count==pass_b_steps`; interpretation of freeze and post-freeze fields
must require this flag. Per-sample derived datasets follow the selected dense
or ragged trace layout.

| Dataset family | Definition |
|---|---|
| `net_linear_fraction`, `net_circular_fraction`, `final_evpa_wrapped_rad` | final observer-basis polarization fractions and EVPA |
| `linear_fraction`, `circular_fraction` | sample `sqrt(Q^2+U^2)/abs(I)` and `V/I` |
| `evpa_wrapped_rad`, `evpa_unwrapped_rad`, `evpa_unwrapped_change_rad` | sample EVPA modulo pi and nearest-period continuation |
| `cumulative_{absorption,faraday_rotation,faraday_conversion,faraday_operator}_depth` | path-ordered cumulative raw depths |
| `{absorption,faraday_rotation,faraday_conversion,faraday_operator}_depth` | final per-ray depths |
| `emissivity_weight`, `cumulative_emissivity_fraction` | `max(j_I,0) abs(dlambda)` and its path CDF |
| `emissivity_formation_radius_{low,median,high}` | radius-sorted empirical 5/50/95-percent quantiles weighted by `max(j_I,0) abs(dlambda)`; not observer weighted |
| `intensity_fraction_of_final` | propagated sample I divided by final propagated I |
| `intensity_freeze_valid`, `intensity_freeze_lambda`, `intensity_freeze_radius` | earliest complete sample after which I stays within 1 percent of the final scale |
| `post_freeze_{absorption,faraday_rotation,faraday_conversion,faraday_operator}_depth` | path-ordered depth from the intensity-freeze sample to the camera |

Radius quantiles sort samples by radius;
older products with `emissivity_formation_definition` containing `in path order`
store the radius at a *path-CDF crossing*, a different quantity. The path CDF
itself retains path ordering. Consult this attribute when reading archived
v2/v3 files rather than inferring semantics from the schema version alone.
The intensity-freeze tolerance now uses the positive final propagated I;
legacy output used the greater of that value and the history's maximum I.
No positive final intensity means no valid freeze sample. Published candidate
traces have been checked against the corrected criterion with unchanged indices.

Freeze validity additionally needs `reason=1` for a returned observer ray.
Intensity stability can coexist with continuing emission and absorption;
post-freeze depths alone do not prove a passive foreground screen. Check the
remaining emissivity and actual polarization-vector changes as well. An EVPA
history near zero linear amplitude, or with phase changes unresolved by the
sample cadence, does not support a unique count of rotations.

### Single- and multi-frequency layout

In trace schema v2, geometry and frequency-dependent fields coexist directly
under `/trace`. The file contains exactly one observing frequency. Trace v1 is
the same legacy layout without final Stokes and `/derived` products.

In trace schema v3, shared geometry is under `/trace/shared`, observing
frequencies are in `/grid/frequency_hz`, and frequency-dependent coefficients
and Stokes values are under `/trace/freq_<fi>`. Each frequency group records
its `freq_index` and `frequency_hz`. All frequency groups must use the same
`/rays/sample_count` and `sample_offset`; this is guaranteed by the shared
step-control path and should be validated by readers. Trace v2 was the
corresponding legacy multi-frequency layout without final Stokes and derived
subgroups.

New multi-frequency traces additionally store `/trace/freq_<fi>/rays` with
the frequency's own final propagated/observer Stokes and ray diagnostics.
The root `/rays` terminal Stokes remain a first-frequency compatibility copy.
A reader must not use that copy to validate another frequency. Legacy files
without per-frequency terminal Stokes must be regenerated before a nonzero
frequency can pass the independent mechanism-replay closure check.

`scripts/analyze_trace_mechanisms.py` reads complete double-precision traces,
independently replays the coupled transfer, and closes both the full history
and final Stokes before reporting mechanisms. It records attenuated, signed V
production by intrinsic emission, selective absorption and conversion, plus
controlled coefficient interventions and subdivision checks. These production
locations complement the image-analysis seed-emission locations.


## Compatibility and release policy

The following changes are backward-compatible within an existing schema
version:

- adding an optional attribute, group, or dataset;
- adding a new termination value, provided old readers treat unknown values as
  invalid;
- adding a new `trace_fields` selector whose absence has no effect on existing
  selectors.

The following changes require a new schema version and an explicit migration
note:

- removing or renaming an existing field;
- changing a field's physical meaning, units, sign convention, dtype, or axis
  order;
- changing invalid-pixel or flux-summing semantics;
- moving single-frequency fields or changing the ragged offset contract.

For such a change, KPolaris must retain a reader for the previous published
version and provide either a deterministic converter or a documented
side-by-side read path. A migration must never overwrite the source file by
default. The release changelog must identify the first software version that
writes the new schema.

The first paper tag freezes `kpolaris_image` v1,
`kpolaris_physical_analysis` v2, single-frequency trace v2, and
multi-frequency trace v3. Pre-tag development products with physical-analysis
v1, trace v1, multi-frequency trace v2, or missing current provenance fields
may be inspected, but they must not be mixed silently with the final paper
validation archive.

## Minimal h5py examples

Read a physical image and mask invalid rays:

```python
import h5py
import numpy as np

with h5py.File("image.h5", "r") as h5:
    if h5.attrs["schema"] != "kpolaris_image" or h5.attrs["schema_version"] != 1:
        raise ValueError("unsupported KPolaris image schema")
    group = h5["/frame_0/freq_0"]
    valid = h5["/frame_0/diagnostics/reason"][...] == 1
    I_nu = group["I"][...]
    I_valid = np.where(valid, I_nu, np.nan)
    flux_I_Jy = group.attrs["flux_I_Jy"]
```

Iterate over a ragged trace:

```python
with h5py.File("trace.h5", "r") as h5:
    if h5.attrs["schema"] != "kpolaris_trace":
        raise ValueError("not a KPolaris trace")
    offsets = h5["/rays/sample_offset"][...]
    nfreq = int(h5.attrs.get("nfreq", 1))
    trace = h5["/trace"] if nfreq == 1 else h5["/trace/shared"]
    radius = trace["r"]
    for ray in range(len(offsets) - 1):
        r_ray = radius[offsets[ray]:offsets[ray + 1]]
```

## Binary trajectory input schema: `kpolaris.binary_trajectory.v1`

The dynamical Superposed Kerr--Schild metric consumes a separate trajectory
input file. Its authoritative marker is on `/trajectory`, not at the root:

```text
/trajectory.attrs["schema"] = "kpolaris.binary_trajectory.v1"
/trajectory.attrs["schema_version"] = 1
```

The other required group attributes are `units="G=c=M_ref=1"`,
`position_gauge="harmonic"`, `spin_convention="kerr_a"`, `generator`,
`trajectory_model`, `pn_terms`, `worldline_velocity_consistent`,
`boost_velocity_model`, and an `interpolation` value of
either `linear` or `cubic_hermite_position_velocity`. The latter requires
`worldline_velocity_consistent=1` and
`boost_velocity_model="derivative_of_position"`; it evaluates position and
velocity from one coupled cubic Hermite polynomial, so velocity is the
analytic time derivative of the interpolated position. The paper-faithful
independent velocity blend instead uses `interpolation="linear"`,
`worldline_velocity_consistent=0`, and
`boost_velocity_model="paper_eq16_independent_blend"`. Readers
reject an unknown schema/version, inconsistent interpolation metadata,
missing required semantics, nonfinite value, non-increasing time grid,
superluminal boost, or invalid
exact-remnant endpoint rather than guessing a repair.
If optional `/trajectory/spin_chi` is present, readers additionally enforce
`kerr_a = mass * spin_chi` component by component. The
`paper_cbwaves_4pn_local` runtime selector is stricter than generic `table`:
it accepts only a native file with the exact model, PN mask, generator,
generator version `1.3`, completion contract
`selected_merger_state_fixed_tolerance_v1`, `source_verified=1`, DOI,
source-verification label, and exact upstream and
patched CBwaves hashes emitted after validating the pinned Zenodo source. It
also requires `merger_separation_reached=1` and
`trajectory_status="merger_separation_reached"`. A hash-verified but
`diagnostic_truncated` file remains usable only through the generic `table`
selector and must not be presented as a completed paper inspiral-to-merger
trajectory.
The completion flag is evaluated at the actual selected merger-time state,
using the file's declared position interpolation and a fixed numerical
tolerance independent of output cadence. Root diagnostics record
`selected_merger_separation`, `merger_reach_evaluation_time`,
`merger_reach_interpolation`, `merger_separation_numerical_tolerance`, and the
acceptance threshold. The contract/version gate intentionally prevents files
created by an older cadence-dependent completion test from passing the strict
selector; those files remain available through generic `table` mode.

Required IEEE numeric datasets are:

| Path | Shape | Unit | Meaning |
|---|---:|---|---|
| `/trajectory/t` | `(Nt)` | `M_ref` | strictly increasing table time |
| `/trajectory/mass` | `(Nt,2)` | `M_ref` | masses of the two SKS terms |
| `/trajectory/position` | `(Nt,2,3)` | `M_ref` | global Cartesian term centers |
| `/trajectory/velocity` | `(Nt,2,3)` | `c` | Lorentz-boost parameters |
| `/trajectory/kerr_a` | `(Nt,2,3)` | `M_ref` | dimensional Kerr vectors `S/M` used by the metric |
| `/trajectory/merger_weight` | `(Nt)` | 1 | smooth merger window `W` in `[0,1]` |

A native file carrying the strict paper completion contract must also contain
scalar `/remnant/transition_start` and `/remnant/transition_end`. Both values
must be explicit trajectory-time samples: `W(start)=0`, the next sample has
`W>0`, `W(end)=1`, the preceding sample has `W<1`, and `end` is the first
exact-unity sample. Missing, one-sided, off-grid, or weight-inconsistent
endpoints are rejected rather than inferred.

Axis 1 is `[BH1 (heavier/equal), BH2]`; axis 2 is `[x,y,z]`. The canonical
mass ratio metadata use `q=m2/m1` in `(0,1]`. Root hard links named `time`,
`mass`, `position`, `velocity`, `spin_a`, and `window` may be written for
interactive convenience, but they are not a replacement for the required
`/trajectory` datasets.

`/trajectory/velocity` is deliberately named a boost velocity. In the
`paper-blend` mode it implements Combi--Ressler v3 Eq. (16) and need not equal
`d(position)/dt` during the algebraic merger transition. In the `consistent`
mode it is the analytic position derivative, including the `dW/dt` collapse
term; the generator must widen or reject a transition that violates the
configured subluminal bound. Runtime interpolation in this mode couples
position and velocity through cubic Hermite interpolation. Mass, `kerr_a`, and
`merger_weight` remain linearly interpolated in both modes. The two velocity
semantics must be distinguished by `worldline_velocity_consistent`,
`boost_velocity_model`, and the root `velocity_semantics` attribute. They agree
with the center derivative in the inspiral and exact postmerger regions.

At every sample with `W=1`, a valid exact-collapse tail has coincident
positions, identical velocities, identical `kerr_a`, and two masses equal to
`Mf/2`. The two rank-one SKS terms then add algebraically to one mass-`Mf` Kerr
metric. An optional `/trajectory/spin_chi` is diagnostic only: because each
half-mass term carries the full remnant `kerr_a`, its per-term `a/m` can exceed
one and must not be used to construct two independent horizon radii.

The reader may analytically continue a verified `W=1` endpoint as a uniformly
moving single Kerr remnant. All other requests outside the table are errors;
clamping to the first/last inspiral sample is forbidden. The legacy importer
for flat Combi--Ressler files is an explicit compatibility path and records
warnings/repairs; it does not change the native contract above.

See [the model and trajectory guide](user_guide.md) for the input workflow.


## Physical parameter responses and source partitions

An optional `analysis/physical_response` group (schema version 1) extends the
analysis products without changing existing fields. It is enabled by
`analysis_partition` or `analysis_response`. See
[physical response diagnostics](diagnostics.md) for the
parameter families, partition definitions, numerical method, and validation.

`source/{I,Q,U,V}_inv` stores observer-basis source contributions with shape
`[bin, ny, nx]`. Under `derivative/{emission,absorption,rotation,conversion}`,
`d{I,Q,U,V}_inv_dlogp` stores signed local-mechanism responses in the same shape.
`reruns/{plus_h,minus_h,plus_half_h,minus_half_h}/{I,Q,U,V}_inv` contains four
complete perturbed transfers on the baseline sample grid. These have the same
`[ny,nx]` shape and invariant units as the image. Derivatives use the logarithm
of the declared parameter scale at scale one. `response_available=0` indicates
source-only products; derivative and rerun groups are then absent.

The group records `parameter`, `log_parameter_step`, `partition`, `bins`,
`partition_edges` where relevant, and physical/numerical definitions in its
attributes. A scalar partition includes explicit underflow, overflow, and
unavailable bins. Partition labels and masks belong to the baseline model.
Source Stokes close linearly, while local finite-difference mechanism responses
converge to the complete derivative with O(h^2) error. Use the archived reruns
to verify that convergence before interpreting derivative ratios.

## Optional direct-only image selection

Root `direct_only` (0/1) and `emission_segment` (`all` or `n0_vertical_turn`)
record the emission selection. When enabled, `direct_only_reference`,
`direct_only_boundary`, `direct_only_transfer`, and `direct_only_no_turn`
state the reference, observer-to-source first vertical-turn convention,
zero incident Stokes equivalence, and no-turn behavior. All analysis groups
then refer to the retained segment. See [diagnostics.md](diagnostics.md).


## Optional equatorial source selection

Root attributes `equatorial_h_over_r` (float64, default 0),
`equatorial_samples` (int32, default 8), and `faraday_rotation` (int32,
default 1) are also stored under `/parameters/radiation`.
`equatorial_emission_definition`, `equatorial_transfer_definition`, and
`equatorial_sampling` give the coordinate, coefficient and integration
conventions; positive width additionally records `equatorial_reference`.
These are additive metadata; the image schema version remains 1.

The wedge keeps volume emission where `abs(cos(theta_BL))=abs(z_KS)/r_BL`
is no greater than its half-height ratio. Zero disables the selection and
one retains all emission. Only exterior emissivities are removed. Absorption
and conversion stay active throughout the configured plasma domain;
`faraday_rotation=0` independently zeroes rho_V, including in parameter reruns.
Analysis emission weights/source tags reflect the selected source; propagation
depths reflect the retained foreground. See [the guide](diagnostics.md).

Observable-response exports (`kpolaris_observable_response_v1`) also copy the
emission-selection parameters and coordinate/transfer definitions to root
attributes. Physical-response JSON and gallery summaries retain the same
selection metadata. A gallery requires identical selection/Faraday settings
as well as a matching baseline image across its parameter products.

## KHARMA DDC input metadata

Image products record the effective stepping value in `slow_light_step_mode`:
`decoupled` (cache-independent integration), `block` (clip at resident block
ends), or `snapshot` (clip at every snapshot). `decoupled` is the default for
all supported GRMHD image backends, including multiple frequencies, diagnostics,
and observer-time batches. These workflows use the same resident-data scheduler;
there is no implicit fallback to snapshot stepping. The input alias `continuous`
is always saved as `decoupled`. A geometric time probe does not perform radiation
transfer and ignores this policy. Fast-light images do not use this setting.

The mode is independent of temporal interpolation. Retain the window count and
input time list for legacy `block` replay, whose partition depends on residency.
Old parameter files omitting this option adopt the current default; select
`block` or `snapshot` explicitly to reproduce the corresponding former policy.

The independent root string attribute `slow_light_interpolation` records
`fluid` (default) or `coefficients`. The effective parameter file retains the
same choice; trace products also record it in `parameters/integration`.
`coefficients` interpolates the eleven transfer coefficients. `fluid`
interpolates sampled fluid quantities, reconstructs four-vectors and applies
the plasma cutoff before evaluating coefficients; existing spatial and
derived-scalar reconstruction conventions are retained. Older products
without this attribute used coefficient interpolation. The choice is inactive
when `slow_light=0` and does not change the Stokes datasets or schema version.

KHARMA image and trace products record `kharma_ddc_native`,
`kharma_ddc_socket`, `kharma_ddc_manifest`, `kharma_ddc_timeout_seconds`,
and `kharma_ddc_protocol` at the root and in `parameters/radiation`.
These describe the resolved input configuration; protocol is `STAGE1` when
native input is enabled and `disabled` otherwise. Ordinary filenames in a
mixed input list still use PHDF. The manifest is declared by the caller, not
authenticated by STAGE1. Archive the actual decoder manifest and its hash
alongside the result. Existing Stokes/analysis/trace datasets are unchanged.
See [DDC input](slow_light.md) for sequence names and time semantics.
