# Changelog

## [0.1.0] - 2026-09-21

Initial distribution of KPolaris:

- Polarized GRRT on CPUs and GPUs through Kokkos.
- Analytic RIAF and torus models; iHARM/HARM, KHARMA, AthenaK and BHAC inputs.
- Fast-light and slow-light images, shared-geometry frequency groups and observer-time batches.
- Cache-decoupled slow-light stepping with fluid interpolation; optional coefficient interpolation.
- Explicit camera, Stokes and EVPA conventions with `evpa_0=N|W` outputs.
- Emission-source, propagation and physical-parameter response diagnostics.
- Direct-only and equatorial emission selection, and sampled ray traces.
- Optional KHARMA DDC input using a separately installed decoder.
- HDF5 outputs with effective parameters, units and build provenance.
- A data-free, 256² M87* analytic first-image example, plotting tools and an agent usage guide.
- Bilingual introductory documentation, a detailed versioned manual and a Wiki exporter.
- Numerical, input/output, installation and workflow regression tests.
