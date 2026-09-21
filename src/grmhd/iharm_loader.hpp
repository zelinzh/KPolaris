#pragma once

#include <string>

#include "common/types.hpp"
#include "model/grmhd.hpp"

namespace kpolaris {

struct IHARMLoadOptions {
    std::string dump_path;
    DefaultReal freq = DefaultReal(230.0e9);
    DefaultReal M_unit = DefaultReal(3.0e25);
    DefaultReal mbh_solar = DefaultReal(6.2e9);
    DefaultReal trat_small = DefaultReal(1);
    DefaultReal trat_large = DefaultReal(20);
    DefaultReal beta_crit = DefaultReal(1);
    DefaultReal sigma_cut = DefaultReal(1);
    DefaultReal sigma_cut_high = DefaultReal(-1);
    int emission_type = 4;
    DefaultReal nonthermal_kappa = DefaultReal(3.5);
    int variable_kappa = 0;
    DefaultReal variable_kappa_min = DefaultReal(3.1);
    DefaultReal variable_kappa_interp_start = DefaultReal(1e20);
    DefaultReal variable_kappa_max = DefaultReal(7.0);
    DefaultReal powerlaw_p = DefaultReal(3.25);
    DefaultReal powerlaw_eta = DefaultReal(0.02);
    DefaultReal powerlaw_gamma_min = DefaultReal(1e2);
    DefaultReal powerlaw_gamma_max = DefaultReal(1e5);
    DefaultReal powerlaw_gamma_cutoff = DefaultReal(1e10);
    int interpolate_derived_scalars = 1;
    int resample_spherical_ks_precomputed = 0;
    int resample_spherical_ks_primitives = 0;
    int resample_n1 = 0;
    int resample_n2 = 0;
    int resample_n3 = 0;
    DefaultReal resample_r_in = DefaultReal(-1);
    DefaultReal resample_r_out = DefaultReal(-1);
    int timing = 0;
};

DefaultReal read_iharm_dump_time(const std::string& dump_path);
GRMHDRadiationModel<DefaultReal> load_iharm_model_from_hdf5(const IHARMLoadOptions& opt);

} // namespace kpolaris
