# The M87* first-image example

The default `quickstart` produces a 256 × 256 polarized image of an analytic
radiatively inefficient accretion flow (RIAF), at 230 GHz. It uses M87*'s mass
and distance. The first figure shows total intensity: an asymmetric emission
ring surrounding a central brightness depression. No simulation files are needed.

## Run and view

From the source directory, with the dependencies installed:

```bash
python3 scripts/kpolaris.py quickstart --output-dir=outputs/first-image
```

Open `outputs/first-image/image_freq0.png`. The complete Stokes I/Q/U/V arrays
are in `image.h5`; `image.h5.params` records the effective parameters, and
`run.json` records the commands and image checks. A successful run has `ok=true`,
finite Stokes and every ray returned. To check installation more quickly, add
`--resolution=64`; the default 256² image resolves the ring much more clearly.

After the initial build, `demo` reuses the executable. The multifrequency,
diagnostics and response examples use the same physical model. A smaller
explicit resolution reduces the cost of checking the more expensive workflows.

## Source and plasma parameters

The authoritative settings are in `params/demo_riaf.par`:

| Quantity | Setting |
| --- | --- |
| Black-hole mass | 6.5 × 10⁹ solar masses |
| Distance | 16.8 × 10⁶ pc (16.8 Mpc) |
| Observing frequency | 230 GHz |
| Dimensionless spin | 0.9375 |
| Observer inclination | 163° from the positive spin axis |
| Field of view | 20 × 20 GM/c², about 76 × 76 microarcseconds |
| Density normalization | `riaf_ne_unit=2e5` cm⁻³, `riaf_nth0=1` |
| Electron-temperature normalization | `riaf_te_unit=5e11` K, `riaf_Te0=1` |
| Radial exponents | Density −1.1, electron temperature −0.84 |
| Vertical scale | `riaf_disk_h=0.35` |
| Velocity prescription | `riaf_keplerian_factor=1`, `riaf_infall_factor=0` |
| Radiating outer radius | 100 GM/c² |

The mass and distance follow [EHT Collaboration 2019, Paper I](https://doi.org/10.3847/2041-8213/ab0ec7).
The corresponding angular gravitational radius is about 3.82 microarcseconds.
The illustrative model has a ring on roughly the 40-microarcsecond scale and
an integrated 230 GHz flux of about 0.6 Jy. These are useful checks of source
units and morphology; this example is not a fit to EHT visibilities or polarization.
Spin and plasma parameters are model choices, not measured properties inferred
from this demonstration.

With radius `r` in GM/c², the density is proportional to
`r^(-1.1) exp[-cot²(theta)/(2 h²)]`, and the electron temperature to `r^(-0.84)`.
The table gives their profile normalizations, not uniform values throughout
the flow. The magnetic strength follows the RIAF model's density-dependent
prescription, with a toroidal field. The velocity prescription uses circular rotation outside the innermost stable
circular orbit and plunging motion inside it. Thermal synchrotron
emission, absorption and Faraday effects are included in the Stokes calculation.

## Read the image

- **The central dark region** reflects photon capture and the distribution of
  the surrounding emission. Its edge is not a drawing of the event horizon.
- **The bright ring** is produced by emitting plasma close to the black hole,
  gravitational lensing and the longer paths of rays near the critical curve.
- **The brightness asymmetry** includes relativistic Doppler beaming from the
  moving plasma. It arises in the transfer solution.

These are the standard physical features discussed in
[EHT Collaboration 2019, Paper V](https://doi.org/10.3847/2041-8213/ab0f43).
The resolved model also contains finer structure than an EHT reconstructed image.
No observing beam or interferometric reconstruction is applied in this example.

Axes show image-plane angular offsets in microarcseconds. They retain the
native camera orientation, rather than claiming celestial north/east. The
inclination of 163° places the observer 17° from the negative spin axis; it is
set in ray initialization. No post-processing reflection or rotation is used.
A sky position angle would require a separately specified physical orientation.

The linear colorbar displays Rayleigh-Jeans brightness temperature,
`T_b = c² I_nu / (2 k_B nu²)`, in 10⁹ K. This is a unit conversion of the observed
specific intensity, not the local electron temperature. All valid intensities
are displayed from zero to the maximum, without logarithmic scaling or clipping
bright pixels. The HDF5 values remain in their original physical units.

## Replot or inspect polarization

```bash
python3 scripts/kpolaris.py inspect outputs/first-image/image.h5 --json
python3 scripts/plot_kpolaris_pol.py outputs/first-image/image.h5 \
  --layout=intensity --intensity-unit=brightness-temperature --fov-units=muas \
  '--title=M87* · analytic RIAF' --output=outputs/first-image/intensity.pdf
python3 scripts/plot_kpolaris_pol.py outputs/first-image/image.h5 \
  --layout=polarization --fov-units=muas --output=outputs/first-image/polarization.png
python3 scripts/plot_kpolaris_pol.py outputs/first-image/image.h5 \
  --layout=stokes --fov-units=muas --output=outputs/first-image/stokes.png
```

The ordered analytic field makes this a useful polarization example, but its
polarization fractions should not be taken as an M87* observational prediction.
For scientific observables, check image resolution, integration tolerances and
the physical model. See [plotting](plotting.md) and
[the orientation conventions](polarization_conventions.md).
