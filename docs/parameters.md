# Solver parameter reference

This page describes the runtime settings for `kpolaris_model_image_<model>` and
additional settings for the trace tools. It covers the public RIAF, binary RIAF,
torus, iHARM, KHARMA, AthenaK and BHAC workflows. Build selections such as
`--models`, `--coordinates` and `--jobs` belong to the [Python helper](cli.md),
not to a solver parameter file. Plotting options are described [separately](plotting.md).

Jump to [camera](#camera-and-field-of-view), [accuracy](#spacetime-domain-and-accuracy),
[radiation](#frequencies-and-electron-distributions), [RIAF](#analytic-riaf),
[torus](#analytic-magnetized-torus), [GRMHD](#grmhd-readers-and-physical-units),
[slow light](#slow-light-batching-and-ddc), [diagnostics](#diagnostics-and-emission-selection),
[binary](#binary-riaf-orbit-and-sources), or [trace](#additional-trace-settings).

## Start with the relevant template

The files in `params/` contain commented groups of settings. Their values are
starting examples, **not universal physical models or accuracy guarantees**.
In particular, a filename containing `recommended` does not make its mass unit,
electron temperature or camera appropriate for another simulation.

| Template | Purpose and settings to check first |
| --- | --- |
| `demo_riaf.par` | Data-free M87* example; mass and distance are specified. Adjust resolution or camera before changing its plasma model. |
| `riaf_recommended.par` | Alternative analytic RIAF example with a Galactic-center mass scale; set distance separately for angular units/flux. |
| `torus_recommended.par` | Analytic magnetized torus; check torus structure, physical normalization and camera. |
| `binary_riaf.par` | Approximate binary with two mini-disks; check orbit, reference mass and time, source extents and validity interval. |
| `iharm_recommended.par` | Native iHARM/HARM input; replace the dump path and set mass unit, electron prescription and coordinate backend. |
| `iharm_sks_primitives.par`, `iharm_sks_precomputed.par` | Explicit alternatives using a resampled spherical Kerr–Schild grid; compile `spherical_ks`. |
| `kharma_recommended.par` | KHARMA PHDF example with spherical-KS resampling enabled. For the native FMKS tutorial or fluid-interpolated slow light, override `kharma_resample=none`. |
| `athenak_recommended.par` | Direct Cartesian Kerr–Schild meshblock sampling; check dump path, physical units, electron prescription and domain. |

For a first GRMHD calculation, follow the [input guide](user_guide.md). There is
no bundled BHAC template; its grid-dependent settings are listed below.

## Syntax, precedence and units

```ini
# One setting per line; comments begin with #.
nx=256
ny=256
freq=230e9                 # Hz
parameter_output=auto
```

- Files accept `key=value` (or `key value`). Use unquoted values in files;
  quotes would become part of the value. For paths with spaces, use `key=value`.
- All command-line `--parameter_file=...` files are read in their supplied order;
  later file settings replace earlier ones. Explicit `--key=value` options are
  then applied in command-line order, overriding file settings.
- A `parameter_file` line inside a file does not recursively include another
  file. Supply multiple files on the command line instead.
- `#` starts a comment even inside a value. Empty values are generally rejected;
  `analysis_partition_edges=` explicitly selects its default boundaries.
- Use `0`/`1` for switches. Names are case-sensitive except for explicitly
  supported aliases; for example `riaf_Te0` and `MODEL_M_unit` use capitals.
- Lengths use `GM/c^2`, times use `GM/c^3`, frequencies Hz, source distance pc,
  black-hole mass solar masses, simulation mass unit g. Angles and other units
  are specified below. Binary global coordinates use the total reference mass.
- The geometric integration step is an **affine-parameter increment** with the
  code's photon normalization; it is not a fixed proper distance or snapshot
  time interval. Supply positive step magnitudes; the code sets the direction.

Defaults below describe the parser or automatic behavior. Templates deliberately
override some of them, and input metadata or compiled specializations can resolve
others. Keep the generated `<output>.params` and HDF5 metadata for the actual run.
Changing runtime settings does not require recompilation within the executable's
compiled capabilities.

**Some settings replace related settings, not just the same key.** Setting
`fov` clears earlier `xspan`, `yspan`, `fovy` and angular-field inputs. Thus the
`fov` after `xspan` in `riaf_recommended.par` is the effective field-of-view input.
Setting `max_radiation_depth` assigns both absorption and Faraday limits; place
separate overrides **after** it. Prefer a single field-of-view convention.

## Input, output and execution

| Parameter | Meaning and accepted settings |
| --- | --- |
| `model` | Model/readout implementation; use the model compiled into the selected executable. See the template table. |
| `parameter_file` | Input parameter-file path; repeat on the command line to layer files. |
| `output` | Image destination; default `kpolaris_image.h5`. Choose a new path for each scientific run. |
| `format` | `auto`, `hdf5`, or `csv`; auto uses the extension. HDF5 retains the richer metadata and diagnostics. |
| `parameter_output` | `auto` writes `<output>.params`; `none` disables it; a path writes that file. Default `auto`. |
| `evpa_0` | `N` (default): EVPA from image north toward east. `W`: horizontal reference. Changes Q/U reference axes, not I/V or pixel positions; see [conventions](polarization_conventions.md). |
| `timing` | `1` prints timing breakdowns, `0` disables them (default). |
| `repeat_images` | Number of identical fast-light images after loading the model once; default 1. Does not advance simulation time. Requires `parameter_output=auto` or `none` when greater than 1. |
| `split_transport` | `1` (default) launches endpoint tracing and transport separately; `0` requests the combined path where supported. Does not choose fast versus slow light. |
| `help`, `version` | Command-line `--help` prints usage and exits; `--version` prints build information. These are not physical parameter settings. |

## Camera and field of view

| Parameter | Meaning, units and behavior |
| --- | --- |
| `camera` | `pinhole` (usual default), or `parallel_plane`. The binary executable defaults to and requires `parallel_plane`. |
| `coordinate` | Integration backend: `boyer_lindquist`, `spherical_ks`, `cartesian_ks`, `mks`, `fmks`. Must be compiled and compatible with the reader; RIAF, torus, binary and resampled paths can select their required backend. It does not convert the simulation grid. |
| `nx`, `ny` | Positive image pixel counts; bare parser default 96 each, first-image demo 256 each. |
| `radius` | Observer radius or image-plane-center distance, in geometric length units. Sets finite-distance geometry, not source distance in pc. |
| `inclination_deg`, `inclination_rad` | Polar viewing angle from simulation `+z`, in degrees or radians. Last supplied angle wins; there is no independent camera azimuth/roll option. |
| `dsource` | Physical source distance in pc; non-positive means unspecified. Required for angular-FOV input and absolute flux conversion. |
| `fovx_dsource`, `fovy_dsource` | Full angular widths in microarcseconds. One positive width supplies the other if missing. Converted using `dsource` and the model mass. |
| `xspan`, `yspan` | Half-widths in geometric length units. For pinhole, `fov=2*xspan/radius`; for parallel-plane, these directly size the screen. A missing plane `yspan` follows pixel aspect ratio. |
| `fov`, `fovy` | Full screen extents. For pinhole these are dimensionless screen slopes (approximately angular widths in radians for small fields); physical projected widths are `radius*fov`. For parallel-plane they are geometric widths if spans are absent. Missing `fovy` follows `fov`. |
| `x_offset`, `y_offset` | Additive screen-center offsets: dimensionless slopes for pinhole, geometric lengths for parallel-plane. Defaults 0. They are not automatically microarcseconds. |
| `use_pinhole_pixel_bias` | Enables the optional horizontal sampling offset; default 0 uses pixel centers. No effect for parallel-plane. |
| `pinhole_pixel_bias` | Horizontal offset in pixel units when enabled; default -0.01. This is an explicit sampling adjustment, not an EVPA or image-orientation correction. |

For angular FOV, the physical width in geometric units is proportional to
`dsource / black-hole mass`. Changing the mass while keeping microarcseconds
fixed therefore also changes the physical region shown.

## Spacetime, domain and accuracy

| Parameter | Meaning and effect |
| --- | --- |
| `spin` | Dimensionless Kerr `a/M` in [-1,1]. Analytic single-hole models use this; GRMHD input metadata supply their spin. Binary uses `binary_chi1/2`. |
| `inner_radius` | Inner ray boundary in geometric units. Non-positive selects a model-dependent safe boundary; single-hole automatic behavior starts near `1.05*r_horizon` and respects input/coordinate restrictions. |
| `outer_radius` | Outer transfer/endpoint domain in geometric units. Non-positive selects a model-dependent extent; RIAF uses `1.15*riaf_r_max`, torus `1.15` times its derived outer extent. It does not set image FOV. |
| `step` | Initial positive affine step; with `adaptive=0`, the requested fixed geometric step. Additional boundary/radiation controls can still shorten steps. |
| `adaptive` | `1` (default) uses adaptive RK4 with step doubling, `0` requests fixed-step geometry. |
| `adaptive_tolerance` | Local error control on geometry and transported basis; default `1e-10`. Smaller usually costs more. It is neither an image NMSE target nor a physical-data uncertainty. |
| `min_step`, `max_step` | Positive bounds on adaptive affine-step magnitude, with `max_step >= min_step`. Boundary handling can require a shorter final step. |
| `max_steps` | Positive per-ray work limit for each pass. Reaching it is an incomplete-ray condition, not a valid physical truncation. |
| `max_radiation_step` | Maximum affine step used for radiation sampling in the active region; non-positive removes this particular cap. Does not replace geometry or optical-depth controls. |
| `max_radiation_depth` | Legacy/common dimensionless depth limit; assigning it also sets both separate depth limits. |
| `max_absorption_depth` | Maximum local absorption-operator depth per accepted transfer step; includes polarized absorption, not only scalar optical depth. Negative falls back to the common limit; zero disables this limit. |
| `max_faraday_depth` | Maximum local Faraday-operator depth, including rotation and conversion. Negative falls back to the common limit; zero disables it. Not directly the EVPA angle. |
| `substeps` | Positive legacy radiation-substep setting. Current main image paths control sampling through adaptive geometry and the radiation step/depth limits; increasing this is not the way to refine those paths. |
| `closure_x_warning` | Threshold for return-position closure warnings, in coordinate length units; default 1. |
| `closure_k_warning` | Wavevector closure-warning threshold; default `1e-2`. |
| `frame_error_warning` | Transported-frame constraint-warning threshold; default `1e-2`. |
| `basis_identity_warning` | Returned screen-basis closure-warning threshold; default `1e-2`. |

The four warning settings only change reporting; they do not make the integration
more accurate. Refine geometry, radiative sampling and image resolution separately.
For GRMHD, the grid and time spacing also limit physical resolution.

## Frequencies and electron distributions

| Parameter | Meaning and behavior |
| --- | --- |
| `freq` | Single observer frequency in Hz; default `230e9`. |
| `freq_list` | Comma-separated positive observer frequencies in Hz; has priority over a generated frequency grid and `freq`. |
| `freq_min`, `freq_max`, `nfreq` | Positive endpoints and count for a generated frequency grid; used if no list is supplied. |
| `freq_spacing` | `log` (default) or `linear` spacing of the generated grid. |
| `multifrequency_chunk_size` | Frequencies advanced per group; 0 uses compiled capacity. Larger requests are capped at `KPOLARIS_MAX_FREQUENCIES` and the frequency count. Fast-light analysis uses one frequency at a time. Grouping can affect finite-step results; see [slow light](slow_light.md#multiple-frequencies). |
| `emission_fit` | `symphony`/`pandya`/`thermal` selects thermal type 1; `kappa` type 2; `powerlaw` type 3; `dexter` thermal type 4. Must match compiled coefficient support. |
| `emission_type` | Numeric alternative: 1/2/3/4 as above; 0 leaves the model/build default. Prefer `emission_fit` in handwritten files. |
| `nonthermal_kappa` | Dimensionless electron κ; default 3.5. Distinct from `torus_kappa`. |
| `variable_kappa` | `1` enables the GRMHD local κ prescription based on magnetization/beta; default 0. Not activated by merely setting its bounds. |
| `variable_kappa_min` | Lower bound in the variable-κ prescription; default 3.1. |
| `variable_kappa_interp_start`, `variable_kappa_max` | Control the large-κ transition toward thermal emissivity/absorptivity; defaults `1e20` and 7. These do not expand the valid range of the κ Faraday fits. |
| `powerlaw_p` | Electron power-law index; this implementation requires `p>2`, default 3.25. |
| `powerlaw_eta` | Energy-normalization factor relating nonthermal electrons to local magnetic energy; default 0.02. Not simply a fraction of the thermal electron number. |
| `powerlaw_gamma_min`, `powerlaw_gamma_max` | Electron Lorentz-factor bounds; defaults 100 and `1e5`. Require `gamma_min>=1` and an ordered range. |
| `powerlaw_gamma_cutoff` | Accepted compatibility setting, default `1e10`; currently does not modify the implemented fits. |

Thermal builds do not become nonthermal merely by adding κ or power-law values.
See [compiled support, normalization and fit domains](electron_distributions.md).
A build with single-frequency capacity and no analysis is not a general spectrum
executable; use a suitable build or run individual `freq` values separately.

## Analytic RIAF

Ignoring the temperature floor, the implemented profiles are
`ne = riaf_ne_unit * riaf_nth0 * r^riaf_pow_nth * exp[-z²/(2 R² H²)]` and
`Te = riaf_te_unit * riaf_Te0 * r^riaf_pow_T`, where `H=riaf_disk_h`,
`R=r sin(theta)` and `z=r cos(theta)`. The field magnitude follows the model's
fixed density/radius prescription; there is no runtime `riaf_b_unit` option.

| Parameter | Meaning |
| --- | --- |
| `riaf_r_min`, `riaf_r_max` | Inner/outer radiating radii in geometric units, also subject to the horizon and run-domain masks. Defaults 1 and 100. |
| `riaf_nth0`, `riaf_Te0` | Dimensionless profile normalizations; defaults 1. |
| `riaf_ne_unit` | Electron-density scale in cm^-3; default `5e6`. |
| `riaf_te_unit` | Electron-temperature scale in kelvin; default `1e11`. |
| `riaf_disk_h` | Vertical Gaussian scale `H/R`; default 0.35. |
| `riaf_pow_nth`, `riaf_pow_T` | Radial density/temperature exponents; defaults -1.1 and -0.84. |
| `riaf_mbh_solar` | Black-hole mass in solar masses; default `4.3e6`. Also the total reference mass for the binary model. |
| `riaf_keplerian_factor` | Angular-velocity interpolation: 1 selects the circular/plunging prescription, 0 the free-fall prescription. |
| `riaf_infall_factor` | Radial-infall interpolation toward free fall; default 0. It does not alter the spacetime. |

## Analytic magnetized torus

These settings describe the implemented constant-angular-momentum equilibrium
source; they are not the input parameters of a GRMHD simulation.

| Parameter | Meaning |
| --- | --- |
| `torus_l_lambda` | Sets `l0=l_ms+l_lambda*(l_mb-l_ms)` between the marginally stable/bound values; default 0.78. |
| `torus_wwin` | Potential filling: `W_in=W_c+wwin*(W_cusp-W_c)`; default 1. |
| `torus_kappa` | Polytropic exponent, default 4/3; unrelated to a κ electron distribution. |
| `torus_omegac` | Central enthalpy-density normalization used to construct the pressure constant, default 1. Despite the name, it is **not angular velocity**. |
| `torus_betac` | Gas/magnetic pressure ratio used in the central pressure normalization; default 10. |
| `torus_beta` | Local gas/magnetic pressure ratio in the source model; default 10. Smaller values strengthen the magnetic field at fixed gas pressure and also affect equilibrium quantities. |
| `torus_Rhigh` | Electron-temperature prescription parameter, appearing in the factor `2+Rhigh`; default 1. |
| `torus_bh_mass_solar` | Black-hole mass in solar masses; default `4e6`. |
| `torus_mdot_cgs`, `torus_mdot_code` | Accretion-rate normalizations in g/s and code units. Their ratio together with the time unit sets the mass unit; defaults `1.57e15` and 0.003. |
| `torus_thetae_min` | Floor on dimensionless `Theta_e=kTe/(me*c²)`; default 0.05. |
| `scalar_transport` | Torus-only switch: 0 full Stokes (default), 1 scalar intensity transfer. |

## GRMHD readers and physical units

Replace `MODEL` below with `iharm`, `kharma`, `athenak` or `bhac`. The
model-prefixed forms are preferred, e.g. `kharma_M_unit`; they make scope clear.

| Parameter | Meaning |
| --- | --- |
| `MODEL_dump` | Snapshot path: iHARM/HARM HDF5, KHARMA PHDF, AthenaK binary, or BHAC `.dat`. Native DDC uses a virtual KHARMA frame name instead. |
| `MODEL_M_unit` | Simulation mass normalization in grams. With length unit L, density scales as `M_unit/L³` and field as `sqrt(4*pi*M_unit*c²/L³)` under the code convention. |
| `MODEL_mbh_solar` | Black-hole mass in solar masses, setting L=`GM/c²` and T=`GM/c³`. This is not `M_unit`. |
| `MODEL_trat_small`, `MODEL_trat_large` | Low/high-beta limits of `Ti/Te`: `R=(R_low+R_high*b²)/(1+b²)`, `b=beta/beta_crit`. |
| `MODEL_beta_crit` | Positive plasma-beta transition scale; default 1. |
| `MODEL_sigma_cut` | Magnetization `sigma=b²/rho` threshold; default 1. Above the cutoff the selected plasma ceases to contribute to transfer. |
| `MODEL_sigma_cut_high` | If greater than `sigma_cut`, replaces the sharp edge with the implemented cosine taper to zero. Default -1 retains a sharp cutoff. |

The temperature/normalization defaults differ by reader. Supply values appropriate
to your simulation and science case rather than relying on a bare executable's
defaults. A magnetization cut is a physical modeling choice, not an accuracy knob.

### Native iHARM/KHARMA and resampling

In this subsection `GRID` means `iharm` or `kharma`.

| Parameter | Meaning |
| --- | --- |
| `GRID_interpolate_derived_scalars` | 1 (default) interpolates cell-derived density, temperature, field magnitude, sigma and beta; 0 derives them from sampled primitives where supported. The latter requires `KPOLARIS_IMAGE_REQUIRE_GRMHD_DERIVED_SCALARS=OFF`. Independent of **time** interpolation. |
| `GRID_resample` | `none` (default), `spherical_ks_primitives`, or `spherical_ks_precomputed`; the last two add a spatial interpolation grid. |
| `GRID_resample_n1`, `GRID_resample_n2`, `GRID_resample_n3` | New radial/polar/azimuthal cell counts; 0 chooses the input dimensions. |
| `GRID_resample_r_in`, `GRID_resample_r_out` | New grid's radial bounds in geometric units; non-positive chooses input-dependent bounds. |
| `GRID_resample_spherical_ks_primitives`, `GRID_resample_spherical_ks_precomputed` | Low-level boolean alternatives to `GRID_resample`; prefer the single mode selector to avoid conflicting switches. |
| `kharma_reverse_field` | 1 reverses the physical magnetic field; default 0. The implementation also assigns the BHAC reversal flag, so prefer setting only the active model's options. |

Resampling requires a compiled `spherical_ks` backend and can change numerical
interpolation errors. Native slow light with `fluid` interpolation should use
`GRID_resample=none`. Metadata set the native MKS/FMKS map: its internally stored
`fmks_*` quantities are not arbitrary runtime options in this interface.

### AthenaK

| Parameter | Meaning |
| --- | --- |
| `athenak_gamma` | Gas adiabatic index; non-positive (default -1) uses input metadata. An override changes the thermodynamics inferred from the dump. |
| `athenak_r_in`, `athenak_r_out` | Radial sampling bounds in geometric units; defaults -1 (automatic inner bound) and 1000. Separate from run-level `outer_radius`. |
| `athenak_profile_mode` | Default 0 is physical transfer. Nonzero profiling modes omit sampling/coefficient/frame operations and are not scientific images. Keep 0 for all production calculations. |

### BHAC grid interpretation and cache

| Parameter | Meaning |
| --- | --- |
| `bhac_gamma` | Gas adiabatic index; non-positive (default -1) uses dump metadata. |
| `bhac_r_in`, `bhac_r_out` | Radial model bounds; non-positive (defaults -1) infer them from the grid. |
| `bhac_hslope` | Native polar-coordinate map parameter: `theta=x2+0.5*hslope*sin(2*x2)`; default 0.25 must match the simulation. |
| `bhac_x1_min`, `bhac_x1_max` | Native log-radius coordinate bounds. Defaults 0.17 and 8.1117280833 are example grid values, not universal. |
| `bhac_x2_min`, `bhac_x2_max` | Native polar-coordinate bounds; defaults 0 and pi. |
| `bhac_x3_min`, `bhac_x3_max` | Native azimuthal bounds in radians; defaults 0 and 2*pi. |
| `bhac_nxlone1`, `bhac_nxlone2`, `bhac_nxlone3` | Base-grid cell counts used to interpret the AMR forest; 0 means automatic inference. |
| `bhac_spin_index` | Index in the dump's equation-parameter array containing spin; -1 (default) infers it. |
| `bhac_sfc` | Root-block ordering: 1 (default) selects space-filling-curve ordering, 0 the alternative order. Must reflect the file's layout. |
| `bhac_reverse_field` | Physical field reversal, default 0. |
| `bhac_profile_mode` | Default 0 computes the physical transfer; nonzero values are stage-profiling controls as for AthenaK. |
| `bhac_cache` | Optional path for staged binary input cache. No cache path means no persistent staged cache. |
| `bhac_cache_mode` | `off`, `read`, `write`, or `read_write` (default); governs reuse/creation of that cache. Input and interpretation metadata must match. |

## Slow light, batching and DDC

Slow light needs time-ordered fluid states covering every radiative sample on
the rays. Use the [time-support and batching workflow](slow_light.md); do not
infer the required interval just from the requested observation times.

| Parameter | Meaning and default |
| --- | --- |
| `slow_light` | 0 fast light, 1 time-dependent GRMHD transfer; default 0. Requires slow-light support in the build. |
| `slow_light_observation_time` | Observer arrival time in simulation time units, default 0. Fluid time is `t_obs+x[0]`; this is not the emission time or the first snapshot time. |
| `slow_light_dump_list` | Comma-separated input paths/virtual frame names in strictly increasing physical time order. |
| `slow_light_time_list` | Corresponding comma-separated physical snapshot times; required when they cannot be inferred from the inputs, including native DDC schedules. |
| `slow_light_dump_pattern` | Alternative indexed filename pattern with a printf-style integer conversion such as `%05d`, or a `{}` placeholder; an explicit dump list takes priority. |
| `slow_light_dump_start`, `slow_light_dump_end`, `slow_light_dump_stride` | Inclusive integer filename-index range and positive increment; defaults 0, -1 (unset end), 1. Indices are not physical times. |
| `slow_light_interpolation` | `fluid` (default): time-interpolate sampled fluid quantities and then calculate coefficients. `coefficients`: compute each snapshot's coefficients and then time-interpolate. |
| `slow_light_step_mode` | `decoupled` (default): geometry is independent of cache boundaries. `block` and `snapshot` retain boundary-limited stepping; `continuous` aliases decoupled. Image-workflow option, not a trace option. |
| `slow_light_windows_per_block` | Positive maximum new adjacent intervals in a resident block, default 1. Larger increases resident data and reduces scheduling overhead. |
| `slow_light_snapshot_cache_gib` | KHARMA device snapshot-array budget in GiB; 0 disables automatic sizing. Not a total GPU-memory cap. If a window count is also specified, the tighter limit is used. |
| `slow_light_prefetch` | Enable host prefetch, default 1. |
| `slow_light_prefetch_snapshots` | Host queue capacity in snapshots; 0 chooses it from the window count. Uses host RAM, not the device budget. |
| `slow_light_pipeline` | Overlap KHARMA uploads with integration, default 1. Includes additional resident blocks in memory sizing. |
| `slow_light_batch_jobs` | Text file of observer time, inclusive first/last snapshot-list indices and output path per job. Shares geometry/input among times, with separate Stokes state; see [format](slow_light.md#observer-time-batches). |
| `slow_light_time_probe` | 1 computes geometric time support rather than a Stokes image; default 0. Run separately from observer-time batches. |
| `kharma_ddc_native` | 1 reads native arrays from an external DDC service; default 0 unless set by the environment. |
| `kharma_ddc_socket` | Service socket path; `none` clears it. Required for native DDC. |
| `kharma_ddc_manifest` | Declared archive-manifest path for provenance; `none` clears it. It does not authenticate a server's archive identity. |
| `kharma_ddc_timeout_seconds` | Positive socket timeout in seconds, default 7200. Not an integration time limit. |

DDC defaults may also come from `KPOLARIS_DDC_NATIVE`, `KPOLARIS_DDC_SOCKET`,
`KPOLARIS_DDC_MANIFEST` and `KPOLARIS_DDC_SOCKET_TIMEOUT_SECONDS`; file/CLI values
override them. Compression, server workers and decoder caches are configured
in the separate DDC service, not with GRRT integration tolerances.

## Diagnostics and emission selection

These settings describe outputs or controlled modifications of the source and
transfer. The [diagnostics guide](diagnostics.md) explains their physical meaning
and the distinction between source contributions and parameter responses.

| Parameter | Meaning and default |
| --- | --- |
| `analysis_mode` | 1 enables physical diagnostic maps; default 0. Requires an analysis-enabled build. |
| `analysis_radial_bins` | Radial formation/source bins, default 16; must fit `KPOLARIS_MAX_ANALYSIS_RADIAL_BINS`. |
| `analysis_radial_min`, `analysis_radial_max` | Positive radial-bin bounds in geometric units; non-positive selects automatic model/domain bounds. |
| `analysis_formation_fraction` | Fraction for the reported cumulative formation interval; default 0.9, strictly between 0 and 1. |
| `analysis_response` | `none` (default), `density_scale`, `temperature_scale`, `magnetic_scale`, or `coefficients`; selects the response family, not a finite change to the baseline model. |
| `analysis_response_step` | Logarithmic perturbation h in [1e-6,0.1], default `1e-3`; independent of the ray integration step. Used in coefficient derivatives and matched perturbed reruns. |
| `analysis_partition` | `none`, `radial`, `region`, `plasma_region`, `near_far`, `thetae`, `sigma`, `beta`, `ne_cgs`, or `b_cgs`; labels source locations/perturbation regions. |
| `analysis_partition_edges` | At most 61 strictly increasing positive comma-separated boundaries for scalar partitions; empty selects defaults. Theta_e, sigma and beta are dimensionless, ne in cm^-3, B in gauss. |
| `analysis_funnel_angle_deg`, `analysis_disk_angle_deg` | Polar-angle region boundaries satisfying `0 < funnel < disk < 90`; defaults 20 and 60 degrees. See the region definitions in the diagnostics guide. |
| `analysis_sigma_boundary`, `analysis_beta_boundary` | Plasma-region thresholds, both default 1; labeling does not itself impose a radiation cutoff. |
| `direct_only` | 1 retains emission on the camera-side segment before the first vertical `dz/ds` turning point; default 0. It is a specific emission-selection convention, not a universal photon-ring-order decomposition. |
| `equatorial_h_over_r` | 0 disables the restriction; a value in [1e-6,1] retains emission within `abs(z/r) <= h/r` about the equatorial plane; 1 includes the whole sphere. It is not a coordinate slice or image crop. |
| `equatorial_samples` | Minimum sampling across a monotone crossing of the selected thickness `2h/r`, default 8, accepted range 2–1024. Does not mean eight independent image planes. |
| `faraday_rotation` | 1 retains Faraday rotation (default); 0 sets rho_V to zero. Faraday conversion and absorption are retained. |

Direct-only and equatorial selection refer to a single Kerr spin axis and are
not defined for `binary_riaf`. Responses retain the baseline masks and vary the
specified coefficient/plasma family, as detailed in the diagnostics guide.
Response and partition outputs require `analysis_mode=1` and HDF5 output.

## Binary RIAF orbit and sources

This is an approximate superposed Kerr–Schild spacetime with prescribed sources.
Use it within the [trajectory/source validity contract](hdf5_schema.md#binary-trajectory-input-schema-kpolarisbinary_trajectoryv1).
The `riaf_*` profile settings also apply, with individual-hole coordinates scaled
by each hole's mass and global length/time set by the total reference mass.

| Parameter | Meaning |
| --- | --- |
| `binary_mass_ratio` | Positive `m2/m1`, default 1. |
| `binary_chi1`, `binary_chi2` | Signed aligned dimensionless spins, defaults 0. |
| `binary_reference_separation` | Separation at the reference event in total-mass geometric units; default 20. |
| `binary_reference_phase` | Orbital phase at reference time, radians; default 0. |
| `binary_reference_time` | Reference coordinate time, default 0. |
| `binary_observation_time` | Observer time for the time-dependent binary metric/source, default 0; separate from GRMHD `slow_light_observation_time`. |
| `binary_minimum_separation` | Lower separation bound in the analytic orbit prescription; default 6. Does not turn the approximation into a merger model. |
| `binary_inspiral`, `binary_orbit` | Switch analytic shrinkage and orbital rotation, respectively; defaults 1. |
| `binary_trajectory_model` | `leading_quadrupole` (default) uses the internal orbit; `table` uses an external validated table; `paper_cbwaves_4pn_local` additionally requires that generator's verified provenance contract. |
| `binary_trajectory_file` | External trajectory path. |
| `binary_trajectory_format` | `auto`, `native`, or `combi_ressler`; native files use the documented HDF5 schema. |
| `binary_trajectory_interpolation` | `auto` follows the validated file declaration; explicit `linear` or `cubic_hermite_position_velocity` must agree with it. |
| `binary_trajectory_time_offset` | Offset applied to external trajectory time, in total-mass time units; default 0. |
| `binary_trajectory_mass_scale` | Positive reference-mass override for the legacy importer. Leave non-positive (default -1) for normalized native tables. |
| `binary_metric_derivative_step` | Coordinate finite-difference scale used for derivatives of the approximate metric; default `2e-5`. Independent of ray step. |
| `binary_capture_factor` | Multiplier of each horizon radius used for capture, default 1.02. |
| `binary_tidal_fraction` | Mini-disk outer-radius factor relative to each Roche lobe, default 0.8. |
| `binary_taper_start_fraction` | Fraction of the mini-disk outer radius where its outer taper begins, default 0.85. |
| `binary_density_scale1`, `binary_density_scale2` | Per-hole density multipliers, defaults 1; may also affect the prescribed field magnitude. |
| `binary_temperature_scale1`, `binary_temperature_scale2` | Per-hole temperature multipliers, defaults 1. |
| `binary_field_polarity1`, `binary_field_polarity2` | Per-hole physical field sign, +1 or -1; defaults +1. |

## Additional trace settings

Trace executables must be built with `--trace`. Shared camera, plasma and
integration options have the meanings above, but image-only batching, analysis,
selection and cache controls are not automatically trace options. Consult the
trace executable's `--help`; do not pass an entire image `.params` record without
checking its keys. Trace writes HDF5 regardless of `format`.

| Parameter | Meaning and default |
| --- | --- |
| `trace_mode` | `single` (default) for one pixel, `image` for all pixels. |
| `ix`, `iy` | Zero-based pixel indices for a single-ray trace, defaults 0; inside `[0,nx)` and `[0,ny)`. |
| `trace_fields` | Comma-separated stored fields: `lambda`, `coords`, `plasma`, `x`, `k`, `e1`, `e2`, `state`, `coeffs`, `stokes`, or `all`; default `coords,coeffs,stokes`. |
| `trace_precision` | Output floating-point storage: `float` (default) or `double`. Does not change the build's integration precision. |
| `trace_layout` | `ragged` (default) stores variable-length rays compactly; `dense` uses rectangular storage. |
| `trace_compression` | HDF5 gzip level 0–9, default 4; 0 disables compression. Affects file size/writing, not ray physics. |
| `max_trace_samples` | Maximum stored samples per ray, default 2048. Distinct from the integration work limit `max_steps`; inspect truncation flags. |
| `trace_stride` | Store every Nth accepted sample opportunity, default 1. Reduces output sampling, not the accuracy of the underlying integration. |

Fast-light traces support multiple frequencies; slow-light traces currently
accept one frequency and use snapshot-pair advancement. For dataset meanings,
see the [trace schema](hdf5_schema.md).

For compatibility, trace accepts but does not apply `substeps`, the four image
warning thresholds, and iHARM/KHARMA `GRID_resample` / `GRID_resample_*` settings;
its native sampling is not changed by those image-resampling options.
`athenak_resample_n1`, `athenak_resample_n2`, `athenak_resample_n3` are also
accepted but ignored by the direct Cartesian-KS trace path. Image-only controls
that are not recognized instead produce an error.

## Compatibility names

Prefer the canonical names above in new files. Hyphens in option keys normalize
to underscores, but case is otherwise significant. The following aliases do
not introduce additional physical controls:
| Broad setting | Scope |
| --- | --- |
| `dump` | Assigns the dump path for the GRMHD readers; prefer `MODEL_dump`. |
| `M_unit`, `MBH` | Assign GRMHD simulation mass unit and black-hole mass, respectively; they do not set RIAF/torus masses. |
| `trat_small`, `trat_large`, `beta_crit`, `sigma_cut`, `sigma_cut_high` | Assign the corresponding GRMHD physical controls across readers; model-prefixed names avoid ambiguity. |
| `interpolate_derived_scalars` | Assigns the iHARM and KHARMA spatial-sampling switches. |
| `resample`, `resample_spherical_ks_precomputed`, `resample_spherical_ks_primitives` | Assign the corresponding iHARM and KHARMA resampling modes/switches. |
| `resample_n1`, `resample_n2`, `resample_n3` | Assign both iHARM/KHARMA resampling-grid counts. |
| `resample_r_in`, `resample_r_out` | Assign iHARM/KHARMA resampling bounds and AthenaK/BHAC model bounds; using a reader-specific name is clearer. |
| `reverse_field` | Same behavior as `kharma_reverse_field`, including assigning the BHAC reversal flag; not an iHARM switch. |

Other image-option aliases map as follows. They have exactly the units and
behavior of the named canonical option.

| Canonical option | Alternative names |
| --- | --- |
| `parameter_file` | `input`, `input_file`, `params`, `config` |
| `parameter_output` | `params_output`, `effective_parameters`, `effective_parameter_file` |
| `fovy` | `fov_y` |
| `dsource` | `dsource_pc` |
| `fovx_dsource` | `fovx_muas`, `fovx_uas` |
| `fovy_dsource` | `fovy_muas`, `fovy_uas` |
| `max_radiation_step` | `radiation_sampling_interval` |
| `max_radiation_depth` | `radiation_sampling_depth` |
| `max_absorption_depth` | `absorption_sampling_depth` |
| `max_faraday_depth` | `faraday_sampling_depth` |
| `substeps` | `radiation_substeps` |
| `closure_x_warning` | `warn_closure_x` |
| `closure_k_warning` | `warn_closure_k` |
| `frame_error_warning` | `warn_frame_error` |
| `basis_identity_warning` | `warn_basis_identity` |
| `split_transport` | `split_passes`, `separate_passes` |
| `analysis_mode` | `analysis`, `physical_diagnostics` |
| `analysis_radial_bins` | `formation_radial_bins` |
| `analysis_radial_min` | `formation_radial_min` |
| `analysis_radial_max` | `formation_radial_max` |
| `analysis_formation_fraction` | `formation_fraction` |
| `slow_light` | `slowlight` |
| `slow_light_prefetch` | `slowlight_prefetch`, `prefetch` |
| `slow_light_time_probe` | `slowlight_time_probe`, `time_probe` |
| `slow_light_observation_time` | `slow_light_t_obs`, `t_obs` |
| `slow_light_dump_list` | `slow_light_dumps`, `slowlight_dumps` |
| `slow_light_time_list` | `slow_light_times`, `slowlight_times` |
| `slow_light_dump_pattern` | `slow_light_pattern`, `slowlight_pattern` |
| `slow_light_dump_start` | `slow_light_start`, `dump_start` |
| `slow_light_dump_end` | `slow_light_end`, `dump_end` |
| `slow_light_dump_stride` | `slow_light_stride`, `dump_stride` |
| `freq` | `frequency_hz` |
| `freq_list` | `frequency_list`, `frequencies` |
| `freq_min` | `frequency_min` |
| `freq_max` | `frequency_max` |
| `nfreq` | `frequency_count` |
| `freq_spacing` | `frequency_spacing` |
| `multifrequency_chunk_size` | `frequency_chunk_size`, `freq_chunk_size` |
| `riaf_Te0` | `riaf_te0` |
| `riaf_pow_T` | `riaf_pow_t` |
| `binary_mass_ratio` | `binary_q` |
| `binary_reference_separation` | `binary_separation` |
| `binary_reference_phase` | `binary_phase` |
| `binary_observation_time` | `binary_t_obs` |
| `binary_trajectory_model` | `binary_orbit_model` |
| `binary_trajectory_file` | `binary_trajectory` |
| `nonthermal_kappa` | `kappa` |
| `powerlaw_p` | `power_law_p` |
| `powerlaw_eta` | `power_law_eta` |
| `powerlaw_gamma_min` | `power_law_gamma_min` |
| `powerlaw_gamma_max` | `power_law_gamma_max` |
| `powerlaw_gamma_cutoff` | `power_law_gamma_cutoff` |
| `torus_Rhigh` | `torus_rhigh` |
| `kharma_dump` | `phdf` |
| `athenak_dump` | `athenak_bin`, `bin` |
| `bhac_dump` | `bhac_dat`, `dat` |
| `iharm_M_unit` | `iharm_m_unit` |
| `kharma_M_unit` | `kharma_m_unit` |
| `athenak_M_unit` | `athenak_m_unit` |
| `bhac_M_unit` | `bhac_m_unit` |
| `athenak_r_in` | `athenak_resample_r_in` |
| `athenak_r_out` | `athenak_resample_r_out` |
| `bhac_r_in` | `bhac_resample_r_in` |
| `bhac_r_out` | `bhac_resample_r_out` |
| `bhac_nxlone1` | `bhac_nxlone_1` |
| `bhac_nxlone2` | `bhac_nxlone_2` |
| `bhac_nxlone3` | `bhac_nxlone_3` |
| `bhac_spin_index` | `bhac_nspin` |
| `bhac_cache` | `bhac_staged_cache` |
| `bhac_cache_mode` | `bhac_staged_cache_mode` |
| `timing` | `profile` |

Trace has its own alias set. In addition to the shared canonical settings above,
`fields`, `precision`, `layout`, `compression`/`gzip`, `trace_max_samples` and
`sample_stride` alias the respective `trace_fields`, `trace_precision`,
`trace_layout`, `trace_compression`, `max_trace_samples` and `trace_stride` options.
Its schedule aliases include `observation_time`, `dump_list`, `dump_time_list`,
`dump_pattern`, `dump_start`, `dump_end` and `dump_stride`. Image-only scheduling
controls are not accepted merely because a similar trace alias exists.
