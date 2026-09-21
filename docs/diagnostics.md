# Physical diagnostics and emission selection

Enable `analysis_mode=1` to record where radiation is emitted, how it propagates,
and how selected model parameters affect the observed Stokes images. These
measurements answer different questions; their weights and fixed quantities
must be kept explicit. Each frequency has its own analysis state.

## Formation and propagation maps

| Product | Interpretation |
| --- | --- |
| Emission-weighted fluid quantities | Conditions at the sampled emission sites, before foreground transfer weighting |
| Radial observed Stokes contributions | Radiation originating in each radius bin after full foreground transfer |
| Formation radii | Quantiles of the corresponding nonnegative radial contribution weights |
| Absorption and Faraday depths | Integrated strength of propagation coefficients along the retained path |
| Source partitions | Observed Stokes attributed to selected spatial or plasma regions |
| Parameter responses | First-order change of observed Stokes when a specified physical parameter is varied |

The implementation accumulates `max(coeffs.jI,0)*abs(dlambda)` for local emission
weighting, where `coeffs.jI` is the source coefficient in the code's invariant
transfer equation. It is not an unconverted fluid-frame emissivity or a proper
length element. The active model's normalization must be retained. See the
[field definitions](hdf5_schema.md) before assigning physical units to a stored
weight or affine path length.

Observed radial contributions include absorption, dichroism, Faraday rotation
and conversion in the foreground. Sum their signed I/Q/U/V first, then compute
polarization fractions or EVPA. A region's V contribution can arise from linear
polarization emitted there and converted elsewhere. Large Faraday depth alone
does not establish that a location controls the final polarization.

```bash
python3 scripts/kpolaris.py demo --example=diagnostics --output-dir=outputs/diagnostics
python3 scripts/plot_analysis_maps.py outputs/diagnostics/diagnostics.h5 \
  --fields=observer_weighted_radius_I,intensity_formation_radius_median,absorption_depth,faraday_rotation_depth \
  --output=outputs/diagnostics/formation.png
```

`dominant_*` fields select the single largest-weight sample and therefore depend
on the integration partition. `photon_ring_winding_estimate` is a coordinate
azimuth proxy, not a photon-ring order. RIAF does not define plasma beta/sigma
for these diagnostics: basic maps contain placeholders; plasma source partitions
put unavailable values in a separate bin.

## Physical parameter responses

Each run chooses one family and differentiates with respect to `q=ln(scale)`
about `q=0`. All families keep the metric, camera, velocity, magnetic-field
direction and baseline radiation mask fixed. They recompute emission,
absorption, rotation and conversion, then apply any equatorial selection or
rotation switch used in the baseline.

| `analysis_response` | Perturbation | Physical interpretation |
| --- | --- | --- |
| `density_scale` | `ne -> exp(q) ne`, `B -> exp(q/2) B`; fixed electron temperature, sigma and beta | Changing the mass normalization of a fixed dimensionless GRMHD state |
| `temperature_scale` | `Theta_e -> exp(q) Theta_e` after the baseline prescription and floor | A global electron-temperature scale, not a derivative with respect to `R_high` |
| `magnetic_scale` | `B -> exp(q) B`, `sigma -> exp(2q) sigma`, `beta -> exp(-2q) beta`; fixed electron temperature | Field-strength changes at fixed field geometry |
| `coefficients` | Independent logarithmic amplitudes of four coefficient blocks | Controlled emission/absorption/rotation/conversion perturbations, not fluid-parameter derivatives |
| `none` | No coefficient perturbation | Source contributions only |

These families act on sampled plasma quantities and use the same coefficient
implementation as the baseline. A variable kappa depending on sigma/beta is
reevaluated for the magnetic family. Moving a magnetization exclusion boundary
or changing GRMHD dynamics requires a different model experiment. Rebuilding
an AthenaK float32 derived cache can introduce small quantization differences
relative to changing the sampled state directly.

For transfer written as `dS/dlambda=j-KS`, a parameter derivative satisfies

```text
d(dS/dq)/dlambda = -K(dS/dq) + dj/dq - (dK/dq)S.
```

The implementation differentiates its actual semianalytic Strang transfer step
by local central differences in four coefficient blocks: emission
`jI,jQ,jU,jV`; absorption `aI,aQ,aU,aV`; rotation `rhoV`; conversion `rhoQ,rhoU`.
Each local response then undergoes the full baseline foreground transfer.
The four mechanisms and all partitions sum to the full derivative as the
finite-difference increment converges. At finite increment h, there is an
`O(h²)` error, not a machine-precision derivative identity.

The same calculation also advances four complete perturbed transfers at
`q=+h,-h,+h/2,-h/2`, from their own zero initial Stokes. These independently
evolving states check the mechanism sum and convergence of the parameter
difference. They share the sampled geometric path. Geometric and radiation
sampling convergence must be checked separately.

## Source and response partitions

`analysis_partition` assigns each valid baseline sample to exactly one bin.
Response partitions remain fixed on the baseline model.

| Partition | Definition |
| --- | --- |
| `radial` | Logarithmic radius bins set by `analysis_radial_bins/min/max`; end bins include exterior samples |
| `region` | Polar angle from the nearer axis: by default funnel `<20°`, sheath `20–60°`, disk `>=60°`, each split near/far |
| `plasma_region` | Sigma threshold, then beta threshold for the remainder; includes an unavailable bin |
| `near_far` | Sign of the position direction dotted with the camera direction; a spatial hemisphere label |
| `thetae`, `sigma`, `beta`, `ne_cgs`, `b_cgs` | Bins of the corresponding plasma scalar |

Geometric region names specify angular ranges and do not prove that material
is an unbound jet. Near/far labels are not the foreground/background order along
a bent ray. Source labels identify where emission originated; response labels
identify where the perturbation acted.

Set `analysis_partition_edges` to at most 61 strictly increasing positive
boundaries for scalar partitions. Bins include underflow, overflow and
unavailable values; equality belongs to the upper bin. Default boundaries are
`1,3,10,30,100` for Theta_e, `0.01,0.1,1,10,100` for sigma/beta,
`100,1000,10000,100000,1000000,10000000` for ne in cm^-3, and
`0.1,1,10,100,1000` for B in gauss. Geometry uses
`analysis_funnel_angle_deg` and `analysis_disk_angle_deg`; plasma-region
thresholds use `analysis_sigma_boundary` and `analysis_beta_boundary`.

## Run and validate a response

Use the RIAF quickstart build, or substitute the corresponding GRMHD executable
and a physical parameter file:

```bash
mkdir -p outputs/temperature-response
./build/cpu-quickstart/kpolaris_model_image_riaf \
  --parameter_file=params/demo_riaf.par --analysis_mode=1 \
  --analysis_response=temperature_scale --analysis_response_step=0.0003 \
  --analysis_partition=radial --max_radiation_step=0.25 \
  --output=outputs/temperature-response/image.h5
python3 scripts/analyze_physical_responses.py outputs/temperature-response/image.h5 \
  --output=outputs/temperature-response/report.json \
  --plot=outputs/temperature-response/response.png \
  --maps=outputs/temperature-response/maps.h5
```

`analysis_response_step` is the parameter increment h, independent of ray step
size. A small fractional increase epsilon gives
`Delta S ≈ epsilon*dS/dln(scale)`. Use `--analysis_response=none` for source
tagging alone, or `coefficients` for independent mechanism perturbations.
Postprocessors accept `--frame` and `--freq-index`.

The analysis checks returned rays, source-Stokes closure, agreement with the
complete half-increment central difference, and convergence between h and h/2.
Defaults require source closure below `1e-10` and relative L1 derivative errors
below `1e-2`. Failed checks save a report and return nonzero. Absolute errors
normalized to total I help interpret derivatives near zero. Ratio and angle
responses use validity masks near zero intensity or linear polarization;
absolute Stokes derivatives are retained.

Source residuals are normalized per pixel by the sum of absolute source Stokes.
The denominator is bounded below by float64's smallest normal number
(approximately `2.23e-308`) to avoid one-roundoff-unit failures in subnormal
values. The report records this floor and affected pixels. It does not modify
images or relax checks in the normal floating-point range.

HDF5 `analysis/physical_response` contains source I/Q/U/V by bin, mechanism
derivatives by bin, the four perturbed full-transfer images, and parameter/partition
definitions. See the [schema](hdf5_schema.md). Response arrays cost
`4*8*(5*Nbin+4)` bytes per pixel; source-only arrays cost `4*8*Nbin`. A 256²
response with 16 bins requires 168 MiB for these arrays, in addition to the
model, ordinary image and other diagnostics. Responses also evaluate perturbed
coefficients and are more expensive than ordinary imaging.

## Direct-only emission

`direct_only=1` selects the segment from the observer to the first backward-ray
turn from moving away from the equatorial plane to moving toward it. The
criterion uses `z=r cos(theta)` and the first applicable zero of `dz/ds`.
If no such turn occurs, the ordinary path is retained. This is a specified
vertical-turn convention, not an equatorial-crossing count or a universal
definition of image order.

With zero incident background, setting emission to zero beyond that endpoint
leaves zero Stokes there. Forward transfer can therefore start at the turn with
zero Stokes, retaining all foreground emission, absorption and Faraday effects.
Geodesic bending and slow-light delays are preserved. This implementation would
need modification for a nonzero incident background.

```bash
./build/cpu-quickstart/kpolaris_model_image_riaf --parameter_file=params/demo_riaf.par \
  --direct_only=1 --output=direct.h5
python3 scripts/plot_direct_only_compare.py full.h5 direct.h5 --output=direct_comparison
```

Compute `full.h5` with the same settings and `direct_only=0`. Independent adaptive
runs may sample different points, so their difference has integration error.
Formation and path-depth diagnostics refer to the retained segment. The option
supports single-Kerr image models, multiple frequencies, slow light and responses;
the binary model rejects it. Output records the precise segment definition.

## Finite equatorial emission layer

`equatorial_h_over_r=eta` retains volume emission where
`abs(z_KS)/r_BL=abs(cos(theta_BL)) <= eta`. Here h is the half-thickness relative
to the spin equator and r is the Kerr radial coordinate, not cylindrical radius
or proper height. `eta=0` disables selection; `eta=1` retains all emission.
The allowed nonzero range is `[1e-6,1]` in the normal double-precision build.

Outside the layer, only emission is set to zero; foreground absorption,
dichroism and Faraday effects remain. Independently, `faraday_rotation=0` sets
rho_V to zero throughout the domain while retaining Faraday conversion. No
thickness renormalization is applied, so thinner layers generally emit less flux.
Zero thickness is not a surface-emission model.

```bash
./build/iharm/kpolaris_model_image_iharm --parameter_file=iharm.par \
  --iharm_dump=/path/to/dump.h5 --coordinate=fmks --iharm_resample=none \
  --equatorial_h_over_r=0.01 --equatorial_samples=8 --output=equatorial.h5
```

`equatorial_samples` controls numerical sampling along a ray, not image
resolution or physical thickness. For `q=z_KS/r_BL`, each accepted step checks
`abs(q_mid-q_start)+abs(q_end-q_mid) <= 2*eta/N`, with boundary refinement.
A complete monotonic crossing typically takes at least about N accepted steps;
other geometry and transfer constraints can require more. The allowed integer
range is 2–1024. Compare N=8 and 16 and refine radiation steps as needed.

This selection supports single-Kerr image models and can be combined with
direct-only emission. Image and trace interfaces are distinct: these switches
are not available on the trace tool. Source tags count selected emission;
propagation depths include the retained foreground. Parameter responses apply
the same selection and rotation switches to baseline and perturbed states.

```bash
python3 scripts/plot_equatorial_compare.py full.h5 equatorial.h5 \
  --no-rotation=equatorial_no_rotation.h5 --output=equatorial_comparison
```

The optional third input is a matched run with `faraday_rotation=0`. Comparison
tools check camera, source and selection metadata, use common brightness scales,
and preserve signed residuals without image registration or independent flux
renormalization.

## Convergence

`adaptive_tolerance` and `max_step` govern geometric integration;
`max_radiation_step`, `max_absorption_depth` and `max_faraday_depth` constrain
transfer sampling. The radiation step is a normalized affine increment, not a
fixed radial or proper distance. Geometry closure alone does not ensure Q/U/V
accuracy in a Faraday-thick region.

At fixed physical parameters and field of view, refine resolution, geometry,
radiation sampling and the response increment separately. For each Stokes
component, `NMSE(S)=sum((S-S_ref)^2)/sum(S_ref^2)` requires a nonzero denominator
and an independently converged reference. Convergence of I does not establish
convergence of weak polarization or parameter derivatives. The supplied
`physical_response`, `direct_only` and `equatorial` CTest checks validate their
analytic and integration contracts; your simulation still needs its own
observable-level convergence checks.
