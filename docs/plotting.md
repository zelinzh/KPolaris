# Plotting and inspecting results

The public scripts cover Stokes images, polarization, physical diagnostics,
matched-image comparisons and individual ray traces. They read stored results;
plotting does not change the transport calculation. PNG, PDF and SVG outputs
are supported by Matplotlib. Parent output directories are created as needed
by the image and analysis-map plotters.

## Images and polarization

```bash
python3 scripts/kpolaris.py inspect outputs/quickstart/image.h5 --json
python3 scripts/plot_kpolaris_pol.py outputs/quickstart/image.h5 \
  --layout=intensity --intensity-unit=brightness-temperature --fov-units=muas \
  --output=outputs/figures/intensity.png
python3 scripts/plot_kpolaris_pol.py outputs/quickstart/image.h5 \
  --output=outputs/figures/polarization.pdf
python3 scripts/plot_kpolaris_pol.py outputs/quickstart/image.h5 \
  --layout=stokes --output=outputs/figures/stokes.png
```

Quickstart and the multifrequency demo select `--layout=intensity`: one image
of total intensity, with angular axes and a linear Rayleigh-Jeans brightness
temperature `T_b = c² I_nu / (2 k_B nu²)`, labelled in 10⁹ K. This is a unit
conversion of the calculated specific intensity, not the electron temperature.
It requires the selected frequency in the HDF5 metadata. No beam convolution,
clipping of valid bright pixels, image rotation or reflection is applied.
`--title` changes only the figure title. The HDF5 remains unchanged.

Invoking the plotter without a layout selects the polarization view. Its four
panels show specific intensity, linear polarization fraction,
electric-vector position angle (EVPA), and signed circular polarization
fraction. Fractions are labelled in percent. Intensity uses a linear scale.
The Stokes layout defaults to color limits based on the common peak I.
`--stokes-scale=independent` instead selects each component's own peak amplitude
for its color limits, retaining the same physical units and signed values.
This is useful for displaying weaker polarization structures; compare the
colorbar values when assessing their amplitudes. `--columns=4` places the four
panels in one compact row, as on the README:

```bash
python3 scripts/plot_kpolaris_pol.py outputs/quickstart/image.h5 \
  --layout=stokes --columns=4 --stokes-scale=independent \
  --intensity-unit=brightness-temperature --fov-units=muas \
  --title='M87* · analytic RIAF' --output=outputs/figures/stokes-row.png
```

EVPA is defined modulo
180 degrees as `atan2(U,Q)/2` in the stored N/W basis: N (default) starts
from image north (up) toward east (left); W starts from the horizontal direction.
The plotter reads `evpa_0` and labels the zero point. Optional `--evpa-ticks`
converts the angle to the internal screen and projects through the actual
pinhole chart for off-axis rays. Thus N/W files draw the same physical lines.
`--qu-conv=camera` or `--evpa-conv=camera` explicitly displays W values instead.
Pixel-unit plots also account for rectangular pixel sizes. A real celestial
position angle still requires orienting the model on the sky; see [the coordinate and polarization definitions](polarization_conventions.md).

Pixels whose intensity is at most `--intensity-floor` times the peak (default
0.001) are masked in fractional-polarization panels. EVPA also requires linear
polarized intensity above this threshold. This avoids displaying a polarization
angle where there is no resolved polarized signal. Nonfinite Stokes pixels and
pixels with an unsuccessful termination code are masked and reported by
`inspect`. A missing termination map is reported as unavailable, not as a
successful ray check.

Use `--frame=N --freq-index=N` to select a zero-based image group. Selected
frequency metadata overrides the frequency-zero aliases in the file. Use
`--summary-json=...` to save the integrated Stokes summary. The reported net
linear polarization is `hypot(sum(Q),sum(U))/sum(I)`; it differs from the mean
of the per-pixel polarization fractions. Flux in Jy is reported only when a
valid conversion is stored in the file. A missing conversion is not inferred
from arbitrary source defaults.

Axes default to `--fov-units=M` (`GM/c^2`). `--fov-units=pixel` works without a
physical scale. For angular axes, use `--fov-units=muas` with the stored angular
field of view, or supply the actual `--mass-solar` and `--distance-pc`.
By default the native plotter preserves Stokes and the recorded N/W EVPA zero point. It rejects
the former `EofN/NofW` display switches and `--qu-conv=ipole-native`: file-basis
conversion is separate from drawing a physical electric-vector direction.
Cross-code comparisons must use a documented basis mapping and retain its metadata.
`--scale` explicitly supplies a cgs-to-Jy-per-pixel conversion for the displayed
Stokes values and the reported flux summary. It cannot be combined with
`--intensity-unit=brightness-temperature`, which selects a different unit.
Brightness-temperature plotting keeps the physical integrated flux unchanged.
Neither option modifies the HDF5 file.

## Diagnostics and responses

```bash
python3 scripts/kpolaris.py demo --example=diagnostics \
  --output-dir=outputs/diagnostics
python3 scripts/plot_analysis_maps.py outputs/diagnostics/diagnostics.h5 \
  --fields=observer_weighted_radius_I,intensity_formation_radius_median,absorption_depth,faraday_rotation_depth \
  --output=outputs/figures/formation.pdf
python3 scripts/kpolaris.py demo --example=response \
  --output-dir=outputs/response
```

The diagnostic demo shows where observed intensity is formed and the accumulated
absorption/Faraday depths. The response demo performs separate density,
temperature and magnetic-field perturbations with full coefficient
recalculation. These are local sensitivity experiments about the selected
model, not a new dynamical simulation. Source-only response fields, where
requested, keep the propagation operator fixed. See
[diagnostic definitions](diagnostics.md) before interpreting those fields.

Analysis maps support `--frame` and `--freq-index`. Radius and coherence panels
use linear scales; positive depths can use logarithmic scales and signed depths
can use symmetric logarithmic scales, explicitly marked on the colorbar.
Limits are chosen for viewing each map, not for a quantitative comparison
between separate figures. Compare the stored arrays when that is required.

## Other supplied plotters

| Script | Required input and purpose |
| --- | --- |
| `plot_physical_response_gallery.py` | Directory of the matched response runs produced by the response demo; parameter response and formation maps. |
| `plot_direct_only_compare.py` | Matched full-emission and direct-only images; emission retained by the turning-point selection. |
| `plot_equatorial_compare.py` | Matched full-emission and equatorial-selection images; effect of the spatial emission selection. |
| `plot_trace_diagnostics.py` | Trace HDF5 output; fluid and transfer quantities along selected rays. |
| `plot_stokes_compare.py` | Two Stokes CSV images; image differences with the requested conventions. |

Run each script with `--help` for its input arguments. Comparison tools require
matching image geometry, frequency and physical settings; they check metadata
where available. `compare_stokes_hdf5.py` provides numerical Stokes comparison
metrics, including NMSE. An NMSE value is an image-integrated squared difference,
not a bound on every pixel's relative difference. Weak or near-zero reference
pixels can have a large relative difference while contributing little to NMSE.

Ray tracing and emission selection are optional, more expensive workflows. The
initial demo needs neither external GRMHD data nor stored ray paths. A
simulation-based slow-light demo requires your own time sequence and sufficient
time coverage; see [slow light](slow_light.md).
