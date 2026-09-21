#pragma once

#include <string>

#include "grmhd/bhac_loader.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
using HAMRRadiationModel = BHACAMRRadiationModel<Real>;

struct HAMRLoadOptions {
    std::string dump_path;
    DefaultReal freq = DefaultReal(230.0e9);
    DefaultReal M_unit = DefaultReal(1.0e26);
    DefaultReal mbh_solar = DefaultReal(6.2e9);
    DefaultReal trat_small = DefaultReal(1);
    DefaultReal trat_large = DefaultReal(40);
    DefaultReal beta_crit = DefaultReal(1);
    DefaultReal gamma = DefaultReal(-1);
    DefaultReal sigma_cut = DefaultReal(1);
    DefaultReal sigma_cut_high = DefaultReal(-1);
    int emission_type = 4;
    int profile_mode = 0;
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
    DefaultReal r_in = DefaultReal(-1);
    DefaultReal r_out = DefaultReal(-1);
    DefaultReal hslope = DefaultReal(0.3);
    int reverse_field = 0;
    std::string id_order = "root_slot";
    std::string root_order = "morton";
    int timing = 0;
};

DefaultReal read_hamr_dump_time(const std::string& dump_path);
BHACStagedDump read_hamr_staged_dump(const HAMRLoadOptions& opt);
HAMRRadiationModel<DefaultReal> materialize_hamr_model_from_staged(const BHACStagedDump& staged,
                                                                    const HAMRLoadOptions& opt);
HAMRRadiationModel<DefaultReal> load_hamr_model_from_dump(const HAMRLoadOptions& opt);

} // namespace kpolaris
