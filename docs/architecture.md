# Code organization

KPolaris integrates null geodesics and polarized radiative transfer on CPUs
and GPUs using Kokkos. The image calculation keeps a compact ray state;
storing the full trajectory is optional through the separate trace tools.

| Directory | Responsibility |
| --- | --- |
| `src/common` | Types, constants and numerical helpers. |
| `src/geometry`, `src/geodesic`, `src/camera` | Metrics, camera initialization, geodesic integration and screen-basis transport. |
| `src/model`, `src/grmhd`, `src/radiation` | Fluid models, simulation sampling, emission/absorption/Faraday coefficients and polarized transfer. |
| `src/image`, `src/kernels` | Image orchestration, slow-light windows, analysis accumulation and parallel kernels. |
| `tools` | Command-line options, input/output and model-specific executables. |
| `scripts` | Deployment, input preparation, result inspection, plotting and comparisons. |
| `params` | Small starting parameter files, including a self-contained RIAF demo. |
| `examples` | C++ examples for using individual library components. |
| `tests` | Numerical, input/output, command-line and installation regression tests. |
| `skills/kpolaris` | Optional instructions for coding agents using this repository. |

Start with `src/KPolaris.hpp` for the library interface, `tools/kpolaris_model_image.cpp`
for the image command-line entry, and `src/image/driver.hpp` for the
image driver. Model selection happens at build time, so users can compile only
the readers they need. `src` and `tools` are the same scientific implementation
used by all deployment workflows.

## Photon propagation convention

The stored wavevector is future-directed, along the physical photon propagation
direction. Pass A starts at the camera and uses a negative affine step to locate
the other endpoint. Pass B uses a positive affine step and transports radiation
back to the camera. The wavevector is not negated between passes. The screen
basis remains parallel transported and determines the camera polarization
reference. Frequency is evaluated as `-u_mu k^mu` with this physical wavevector.

This parameterization describes the same reversible vacuum geodesic as a
past-directed ray with the opposite affine parameter. Correct equivalence
requires consistent direction and sign conventions in fluid frequency,
polarization and transfer. Keeping one physical wavevector throughout avoids
a convention change at the pass boundary.

## Fast and slow light

Fast light samples one fluid snapshot. Slow light evaluates the fluid at the
coordinate time along the ray, loading the surrounding snapshots and advancing
rays through resident time windows. By default it interpolates fluid quantities
before recomputing transfer coefficients; interpolation of coefficients remains
an explicit option. Raw HDF5 and DDC input share the transfer path. Input
preparation, resident snapshot count and GPU advancement are separate concerns;
they do not change the physical time coverage required by the rays.

For all supported GRMHD images, spectra, diagnostics and observer-time batches,
the default `decoupled` mode keeps integration steps independent of cache
boundaries. A missing radiation sample pauses the ray without accepting a
partial step; loading its time interval allows the same proposal to be retried.
Four retained snapshots support step reductions. Frequency groups share geometry
and fluid data while each channel retains its own transfer and diagnostic state.
A geometric time probe does not use snapshot-boundary stepping. See the [slow-light guide](slow_light.md)
for memory sizing, legacy `block`/`snapshot` modes and trace behavior.

## Output and optional analysis

Native HDF5 output stores Stokes arrays, parameters, termination maps and build
provenance. Optional analysis adds formation quantities, transfer depths,
emission selections and controlled responses. The stored definitions are part
of the data contract; readers should use the appropriate schema and conventions
rather than infer a physical meaning from an array name alone.

See [the HDF5 contract](hdf5_schema.md) and [diagnostic definitions](diagnostics.md).
The [coordinate and polarization conventions](polarization_conventions.md)
define image axes, Stokes signs, camera EVPA, and explicit comparison transformations.
