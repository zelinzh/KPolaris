# Electron distributions and transfer coefficients

The standard image builds and all introductory examples use thermal electrons.
`emission_fit=thermal` selects the Pandya emissivity fits; `emission_fit=dexter`
selects the Dexter fits. Thermal absorption follows Kirchhoff's law, and thermal
Faraday rotation and conversion are included. Use the same coefficient choice
when comparing calculations.

| Option | Emission and absorption | Faraday terms | Usage |
| --- | --- | --- | --- |
| `thermal` (`emission_type=1`) | Polarized thermal synchrotron | Included | Default RIAF build |
| `dexter` (`emission_type=4`) | Polarized thermal synchrotron | Included | Default GRMHD build |
| `powerlaw` (`emission_type=3`) | Isotropic power-law fits | Not implemented; zero | Optional nonthermal model |
| `kappa` (`emission_type=2`) | Isotropic κ-distribution fits | Fits interpolated over `3.5 <= kappa <= 5` | Experimental polarized-transfer option |

The nonthermal coefficient functions have regression checks against independent
Symphony fit evaluations. Those checks do not establish complete nonthermal
image agreement: they normalize selected coefficient signs individually, rather
than transform the complete polarized transfer operator between screen bases.
In particular, the κ Faraday branch requires a complete basis-consistent
transport comparison before treating it as a validated polarization model.
Use the thermal branches for the validated introductory workflows.

## Build and select a distribution

Image executables specialize the thermal coefficients at compile time by default.
A conflicting runtime `emission_fit` is rejected. Build the appropriate runtime
selection path to enable nonthermal choices:

```bash
python3 scripts/kpolaris.py build --models=riaf --build-dir=build/riaf-distributions \
  --cmake-arg=-DKPOLARIS_RIAF_EMISSION_TYPE=0
```

For GRMHD readers, use `--cmake-arg=-DKPOLARIS_GRMHD_EMISSION_TYPE=0` instead.
Then set `emission_fit=kappa` or `emission_fit=powerlaw` in the model's parameter
file. Selecting a distribution replaces the thermal coefficients; it does not
automatically add a nonthermal component to a thermal population.

## Physical parameters and supported range

`nonthermal_kappa` sets κ. The default width is
`w = (kappa - 3) / kappa * theta_e`. GRMHD models also offer
`variable_kappa=1`, whose local κ depends on magnetization and plasma beta.
Changing the distribution or enabling spatially varying κ requires checking
the local fit conditions, not just the observing frequency.

The κ Faraday functions return zero outside `3.5 <= kappa <= 5`.
This is an implementation boundary, not a physical prediction of vanishing
Faraday effects. The published fits also become inaccurate at
`X_kappa = nu / (nu_c * (w*kappa)^2 * sin(theta))` below approximately 0.1,
or at `nu/nu_c` below approximately 1. Here `nu` is the fluid-frame frequency.
The solver does not automatically enforce all published fit-validity conditions.
The large-κ emissivity/absorptivity transition controlled by
`variable_kappa_interp_start` and `variable_kappa_max` does not supply missing
Faraday coefficients.

`powerlaw_p`, `powerlaw_gamma_min` and `powerlaw_gamma_max` set the power-law
index and Lorentz-factor limits. Its current density normalization uses
`powerlaw_eta` and the local magnetic field and requires `p > 2` and
`gamma_min >= 1`. The fits are synchrotron approximations, not numerical
integrals of arbitrary truncated electron distributions. The accepted
`powerlaw_gamma_cutoff` parameter currently does not modify these fits; it must
not be used to claim an exponential spectral cutoff. Power-law Faraday terms
are omitted even when `faraday_scale` is nonzero.

The formulas and their conditions are described by
[Pandya et al. (2016)](https://arxiv.org/abs/1602.08749),
[Dexter (2016)](https://doi.org/10.1093/mnras/stw1526), and
[Marszewski et al. (2021)](https://arxiv.org/abs/2108.10359).
See also the [polarization conventions](polarization_conventions.md).
