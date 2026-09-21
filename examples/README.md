# Examples

For a first image, use the self-contained command-line demo from the repository
root:

```bash
python3 scripts/kpolaris.py quickstart
python3 scripts/kpolaris.py demo --example=multifrequency --output-dir=outputs/two-frequencies
python3 scripts/kpolaris.py demo --example=diagnostics --output-dir=outputs/diagnostics
python3 scripts/kpolaris.py demo --example=response --output-dir=outputs/response
```

These examples use the M87* parameters in `params/demo_riaf.par` and need no
simulation files. The default resolution is 256²; `--resolution=64` makes a
faster installation check. See [the physical model](../docs/first_image.md). The
response demo runs three physical perturbations and therefore takes longer.
Each example saves its resolved parameters, HDF5 output, figures, logs and a
machine-readable validation record. See [the quickstart guide](../docs/quickstart.md)
and [plotting guide](../docs/plotting.md).

The C++ files in this directory demonstrate individual library components.
Build them with ordinary CMake:

```bash
cmake -S . -B build/examples -DKPOLARIS_BUILD_EXAMPLES=ON
cmake --build build/examples --parallel 2
```

GRMHD parameter files are templates for user-supplied data. Choose the matching
model executable and set the input path and physical units before rendering.
See the [GRMHD workflows](../docs/user_guide.md) for KHARMA, iHARM and AthenaK
build/run examples, and the [helper reference](../docs/cli.md) for model selections.
Slow-light examples additionally require the full time interval sampled by the
rays; a single snapshot cannot test slow light.
