#pragma once

#include <string>
#include <vector>

#include "grmhd/athenak_loader.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct BHACAMRRadiationModel : public AthenaKDirectRadiationModel<Real> {};

enum BHACCacheMode {
    BHACCacheOff = 0,
    BHACCacheRead = 1,
    BHACCacheWrite = 2,
    BHACCacheReadWrite = 3
};

struct BHACLoadOptions {
    std::string dump_path;
    std::string cache_path;
    int cache_mode = BHACCacheOff;
    DefaultReal freq = DefaultReal(230.0e9);
    DefaultReal M_unit = DefaultReal(1.0e18);
    DefaultReal mbh_solar = DefaultReal(4.14e6);
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
    int nxlone1 = 0;
    int nxlone2 = 0;
    int nxlone3 = 0;
    int spin_index = -1;
    DefaultReal x1_min = DefaultReal(0.17);
    DefaultReal x1_max = DefaultReal(8.1117280833);
    DefaultReal x2_min = DefaultReal(0);
    DefaultReal x2_max = DefaultReal(3.141592653589793238462643383279502884);
    DefaultReal x3_min = DefaultReal(0);
    DefaultReal x3_max = DefaultReal(6.283185307179586476925286766559005768);
    DefaultReal hslope = DefaultReal(0.25);
    DefaultReal r_in = DefaultReal(-1);
    DefaultReal r_out = DefaultReal(-1);
    int sfc = 1;
    int reverse_field = 0;
    int timing = 0;
};

struct BHACStagedDump {
    int nblocks = 0;
    int nvar = 0;
    int nx1 = 0;
    int nx2 = 0;
    int nx3 = 0;
    DefaultReal spin = DefaultReal(0.9375);
    DefaultReal gamma = DefaultReal(4.0 / 3.0);
    DefaultReal time = DefaultReal(0);
    DefaultReal startx1 = DefaultReal(0);
    DefaultReal startx2 = DefaultReal(0);
    DefaultReal startx3 = DefaultReal(0);
    DefaultReal stopx1 = DefaultReal(0);
    DefaultReal stopx2 = DefaultReal(0);
    DefaultReal stopx3 = DefaultReal(0);
    DefaultReal r_in = DefaultReal(-1);
    DefaultReal r_out = DefaultReal(-1);
    DefaultReal hslope = DefaultReal(0.25);
    std::vector<DefaultReal> extents;
    std::vector<DefaultReal> cell_geom;
    std::vector<float> prims;
    std::vector<int> levels;
    std::vector<int> index_offsets;
    std::vector<int> index_candidates;
    int index_nx = 0;
    int index_ny = 0;
    int index_nz = 0;
    DefaultReal index_xmin = DefaultReal(0);
    DefaultReal index_xmax = DefaultReal(0);
    DefaultReal index_ymin = DefaultReal(0);
    DefaultReal index_ymax = DefaultReal(0);
    DefaultReal index_zmin = DefaultReal(0);
    DefaultReal index_zmax = DefaultReal(0);
    DefaultReal index_dx = DefaultReal(1);
    DefaultReal index_dy = DefaultReal(1);
    DefaultReal index_dz = DefaultReal(1);
};

struct BHACRawStagedDump {
    int nblocks = 0;
    int nw = 0;
    int nx1 = 0;
    int nx2 = 0;
    int nx3 = 0;
    DefaultReal spin = DefaultReal(0.9375);
    DefaultReal gamma = DefaultReal(4.0 / 3.0);
    DefaultReal time = DefaultReal(0);
    DefaultReal startx1 = DefaultReal(0);
    DefaultReal startx2 = DefaultReal(0);
    DefaultReal startx3 = DefaultReal(0);
    DefaultReal stopx1 = DefaultReal(0);
    DefaultReal stopx2 = DefaultReal(0);
    DefaultReal stopx3 = DefaultReal(0);
    DefaultReal hslope = DefaultReal(0.25);
    std::vector<DefaultReal> extents;
    std::vector<DefaultReal> cell_geom;
    std::vector<double> conserved;
    std::vector<int> levels;
    std::vector<int> index_offsets;
    std::vector<int> index_candidates;
    int index_nx = 0;
    int index_ny = 0;
    int index_nz = 0;
    DefaultReal index_xmin = DefaultReal(0);
    DefaultReal index_xmax = DefaultReal(0);
    DefaultReal index_ymin = DefaultReal(0);
    DefaultReal index_ymax = DefaultReal(0);
    DefaultReal index_zmin = DefaultReal(0);
    DefaultReal index_zmax = DefaultReal(0);
    DefaultReal index_dx = DefaultReal(1);
    DefaultReal index_dy = DefaultReal(1);
    DefaultReal index_dz = DefaultReal(1);
};

DefaultReal read_bhac_dump_time(const std::string& dump_path);
BHACStagedDump read_bhac_staged_dump(const BHACLoadOptions& opt);
BHACRawStagedDump read_bhac_raw_staged_dump(const BHACLoadOptions& opt);
BHACAMRRadiationModel<DefaultReal> materialize_bhac_model_from_staged(const BHACStagedDump& staged,
                                                                       const BHACLoadOptions& opt);
BHACAMRRadiationModel<DefaultReal> materialize_bhac_model_from_raw(const BHACRawStagedDump& raw,
                                                                    const BHACLoadOptions& opt);
BHACAMRRadiationModel<DefaultReal> load_bhac_model_from_dat(const BHACLoadOptions& opt);

} // namespace kpolaris
