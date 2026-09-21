# Contributing

Keep scientific changes separate from plotting and deployment changes when possible.
Describe the affected model, coordinate system and expected physical behavior.
Include the effective parameters and the smallest reproducible case when reporting
an issue; simulation data need not be published to report a build failure.

Build the models affected by a change and run their tests:

```bash
python3 scripts/kpolaris.py build --models=riaf,kharma --coordinates=fmks \
  --build-dir=build/tests --tests
```

Unit tests cover geometry, transport and interpolation. CLI tests use generated
fixtures, exercise HDF5 and plotting interfaces, and check installation and source
archives. Physics changes need relevant convergence/observable checks; a small
smoke image alone does not establish scientific accuracy.

Follow the surrounding C++/Python style, keep public names descriptive, and document
changes to units, polarization conventions, parameter meanings or output schemas.
Keep raw data, generated figures, logs and build directories out of changesets.
Respect the licenses of all dependencies and reference implementations.
