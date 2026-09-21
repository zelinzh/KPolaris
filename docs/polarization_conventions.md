# Camera, image orientation and polarization conventions

KPolaris uses metric signature `(-,+,+,+)`. Image directions and Stokes signs
follow the camera geometry and polarization basis. They are not determined by
fitting another image.

## Coordinates and viewing direction

For a single black hole, inclination is measured from coordinate `+z`. Negative
spin means angular momentum along `-z`; coordinate north is therefore not always
the angular-momentum direction. The pinhole camera has native azimuth `phi=0`;
there is no separate camera-roll parameter.

The pinhole camera at `(r,theta,phi)=(R,i,0)` maps to Cartesian Kerr–Schild
position `(R sin i, a sin i, R cos i)`. The spin-dependent second component is
part of the coordinate transformation. Its observer is normal to ingoing
Kerr–Schild time slices, transformed consistently if using Boyer–Lindquist
coordinates. The central ray has zero polar and azimuthal covariant momentum.
The camera tetrad's third spatial axis points outward; its second is constructed
from increasing native polar coordinate, and its first completes the handedness.

The parallel-plane camera instead uses Cartesian Kerr–Schild center
`(R sin i,0,R cos i)`, propagation direction `n=(sin i,0,cos i)`, horizontal axis
`+e_phi=(0,1,0)` and vertical axis `-e_theta=(-cos i,0,sin i)`. At finite radius,
this projection does not in general select the same rays as the pinhole camera.
For an observer exactly on the axis, a limiting azimuth must define screen north.

## Photon direction and transported screen

The stored wavevector `k` is always the future-directed physical photon
wavevector, with fluid-frame frequency proportional to `-u_mu k^mu > 0`.
Pass A integrates backward from the camera using negative affine steps;
Pass B transports radiation toward the camera using positive steps. Neither
`k` nor the screen basis is reset between passes. The user supplies positive
magnitudes for `step`, `min_step` and `max_step`.

For each ray, `e1,e2` are perpendicular to `k`. In the observer rest frame,
`(e1,e2,n)` is right-handed along photon propagation. The pinhole camera projects
and orthogonalizes its tetrad axes for each ray, then parallel transports this
screen. Final Stokes are transformed using the overlap of the transported and
observer screens. A reflected basis changes V according to the determinant of
that overlap; this is not an arbitrary sign adjustment.

## Pixels and image axes

Pixel index is `iy*nx+ix`; HDF5 arrays are `[y,x]`. Increasing `ix` points right
and increasing `iy` points up. Plotters use `origin='lower'` without transposition
or reflection. In the distant-observer limit, image `+x` is `+e_phi` and image
`+y` is `-e_theta`, the projection of coordinate `+z` onto the screen.

At the center of the pinhole camera, both transported screen axes point opposite
the corresponding image axes because sourceward tracing follows `-k`. This
simultaneous 180-degree reversal leaves Q/U/V and unoriented polarization
segments unchanged. Off-axis rays use their actual projected local screens.

Default sampling is at pixel centers. `use_pinhole_pixel_bias=1` explicitly
enables an offset in pixel-width units; `pinhole_pixel_bias=-0.01` reproduces
a legacy ipole sampling offset. This changes sampled rays, not image orientation.
The default is `use_pinhole_pixel_bias=0`.

## Stokes and EVPA

With real electric field `Re[E exp(-i omega t)]`, the local screen convention is

```text
I = <|E1|² + |E2|²>
Q = <|E1|² - |E2|²>
U =  2 Re<E1 E2*>
V = -2 Im<E1 E2*>
chi = 0.5 atan2(U,Q)  (mod pi)
```

Positive Q denotes linear polarization along `e1`, and positive U denotes
45 degrees from `e1` toward `e2`. For example `(E1,E2)=(1,i)` gives positive V
and an electric vector rotating from `e1` toward `e2` as local time increases.
This explicit definition avoids ambiguity in the viewing direction implicit in
the words left- or right-circular polarization. EVPA is undefined when Q=U=0.

Pure Faraday rotation obeys `dQ/dl=-rho_V U`, `dU/dl=rho_V Q`, hence
`dchi/dl=rho_V/2`. In a medium with emission, absorption and conversion, the
integrated rotation coefficient is not generally the observed change in EVPA.

The local synchrotron basis has `e2` along the projected magnetic field, so
positive optically thin `jQ` denotes an electric vector perpendicular to that
field. The same basis transformation is applied to emission, absorption and
conversion coefficients. Reversing the physical magnetic field is a model
change, not an image-axis convention.

## Output zero point and sky orientation

Images use the astronomical display labels north (up) and east (left). They
do not by themselves determine the simulation's position angle on the real sky.
`evpa_0` selects the output reference axes for both image and trace tools:

| Value | Output relative to the internal camera basis | EVPA zero |
| --- | --- | --- |
| `N` (default) | `(I,-Q,-U,V)` | Image north; EVPA increases toward east |
| `W` | `(I,Q,U,V)` | Horizontal camera reference |

This is a passive 90-degree rotation. It leaves pixels, intensity, V and physical
polarization segments unchanged. It applies to every frequency, integrated flux,
radial/source contribution, response derivative and response rerun. The file
and effective parameter record store the selected value. Historical KPolaris
files with `camera` or the old missing-zero convention are read as W; use
`evpa_0=W` explicitly when reproducing those Q/U values.

Trace samples, their coefficients and `final_propagated_*` retain the parallel
transported basis. Only `final_observed_*` and the final observed EVPA use the
chosen output zero point. Geometry and basis-overlap diagnostics are unchanged.

For a general passive rotation, with
`e1'=cos(psi)e1+sin(psi)e2` and `e2'=-sin(psi)e1+cos(psi)e2`,

```text
Q' = Q cos(2psi) + U sin(2psi)
U' = -Q sin(2psi) + U cos(2psi)
I' = I, V' = V
chi' = chi - psi (mod pi)
```

The angle must come from the geometry of the two bases. Comparing with another
code additionally requires the same camera, projection, pixels, frequency,
units and source prescription. Matching ipole's N/W choice removes the output
reference-axis difference; it does not guarantee matching physical rays.
Transposing a stored `[x,y,component]` array to `[component,y,x]` only reorders
storage and must not be confused with a physical reflection.

Plotters read the file zero point. Polarization segments are transformed back
to the internal screen and projected using the pinhole Jacobian; wide-field
angles cannot be interpreted as planar directions without this projection.
`--qu-conv=camera` or `--evpa-conv=camera` explicitly selects W for displayed
values. No additional unconditional Q/U sign flip should be applied.

## Recorded conventions

Image and trace metadata include `polarization_conventions_version=2`,
`stokes_definition`, `electric_field_phase_convention`, `screen_handedness`,
`evpa_definition`, `camera_azimuth_convention`, `sky_orientation`,
`image_pixel_order`, `image_axis_geometry` and `output_stokes_transform`.
Images also record `image_array_order=y,x` and `image_display_origin=lower`.
See the [HDF5 schema](hdf5_schema.md) for units and the distinction between
intermediate and observed quantities.
