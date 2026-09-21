# Working with KPolaris

Use [skills/kpolaris/SKILL.md](skills/kpolaris/SKILL.md) for deployment and scientific
workflows. The source tree is self-contained except for dependencies and user data.

- `python3 scripts/kpolaris.py doctor --json` checks the current environment.
- `quickstart --dry-run --json` previews a data-free RIAF build and demo.
- `quickstart --output-dir=outputs/first-image --json` executes and validates it.
- `inspect IMAGE.h5 --json` reports finite pixels, ray termination and flux scale.

Select CPU/OpenMP/CUDA according to the user's request; keep builds model-specific.
For a first image, build only RIAF and the chosen execution backend. For GRMHD,
select the requested reader and matching coordinates; do not add every model,
`--coordinates=all`, tests or trace tools unless the task needs them. Explain
before starting that initial CUDA GRMHD compilation can take tens of minutes.
Reuse build directories and executables. See [selective builds](docs/quickstart.md#selective-builds-and-existing-dependencies)
for optional feature switches; retain all features required by the requested workflow.
Do not guess GRMHD inputs, mass units, snapshot times or a missing physical scale.
A demo is successful only when its report has `ok=true` and all rays returned.
Use a new output directory for each run. Read the failing stage's log on error.

The physical photon wavevector remains future directed in both passes; the signed
integration step sets the traversal direction. Plotting preserves native camera
Stokes conventions unless an explicit conversion is requested.

Use CMake/CTest for implementation changes. Keep generated images, logs, build
products and simulation data outside tracked source paths. User documentation is
in `docs/`; reusable CLI examples are described in `examples/README.md`.
