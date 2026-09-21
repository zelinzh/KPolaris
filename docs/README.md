# KPolaris User Manual

KPolaris produces polarized black-hole images from analytic plasma models and
GRMHD simulations. This manual covers setup, physical inputs, numerical controls,
diagnostics and interpretation of the saved results.

Run commands from the source directory unless stated otherwise. This English
manual is shipped with the code. A separate [Chinese guide](zh/README.md) is
available; parameter names and file formats are the same in both languages.

## Choose a workflow

| What you want to do | Start here | Then read |
| --- | --- | --- |
| Install and see a black-hole image | [Installation](quickstart.md) | [M87* example and Stokes plots](first_image.md) |
| Use your own GRMHD snapshot | [Reader, units and run commands](user_guide.md) | [Parameter meanings](parameters.md), [plotting](plotting.md) |
| Process an evolving simulation | [Slow light and observer-time batches](slow_light.md) | [Integration and memory controls](parameters.md#slow-light-batching-and-ddc) |
| Study image formation and polarization | [Physical diagnostics](diagnostics.md) | [Polarization conventions](polarization_conventions.md), [HDF5 fields](hdf5_schema.md) |
| Automate runs or build on the code | [Command-line helper](cli.md) | [Code organization](architecture.md) |

Start with the reader and features you need. The first CUDA compilation can
take tens of minutes; the resulting executable is reused for subsequent images.
The [selective-build guide](quickstart.md#selective-builds-and-existing-dependencies)
explains how to limit compilation work.

## Getting started

- [Installation and first image](quickstart.md)
- [Command-line helper: `scripts/kpolaris.py`](cli.md)
- [GRMHD inputs, units and image workflows](user_guide.md)
- [M87* example: physical scales and interpretation](first_image.md)
- [Plotting and inspecting results](plotting.md)

## Scientific use

- [Camera, Stokes and EVPA conventions](polarization_conventions.md)
- [Electron distributions and supported coefficients](electron_distributions.md)
- [Physical diagnostics and emission selection](diagnostics.md)
- [Slow light, batches and optional DDC input](slow_light.md)
- [HDF5 output and trajectory schema](hdf5_schema.md)

## Reference

- [Solver parameters: meanings, units, defaults and scope](parameters.md)
- [Code organization](architecture.md)
- [Versioned documentation and GitHub Wiki](wiki.md)

Simulation data and optional external decoders are supplied separately. Use
the parameter templates shipped with your code version, set the physical units
for your simulation, and check convergence of the observables you intend to use.
