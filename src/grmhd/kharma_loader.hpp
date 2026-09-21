#pragma once
#include "grmhd/ddc_host_buffer.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/types.hpp"
#include "grmhd/ddc_input_options.hpp"
#include "model/grmhd.hpp"

#ifdef KPOLARIS_DDC_COMPACT_CUDA
namespace ddc_transport { struct CompactFrame; }
#endif
namespace kpolaris {

struct KHARMALoadOptions {
    std::string dump_path;
    DDCInputOptions ddc = DDCInputOptions::from_environment();
    // Optional slow-light schedule check against the time carried by STAGE1.
    DefaultReal ddc_expected_time = std::numeric_limits<DefaultReal>::quiet_NaN();
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
    int reverse_field = 0;
    int timing = 0;
};

struct KHARMAStagedDump {
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    std::shared_ptr<ddc_transport::CompactFrame> compact;
#endif
    GRMHDRadiationModel<DefaultReal> model;
    size_t num_meshblocks = 0;
    int nx1_mb = 0;
    int nx2_mb = 0;
    int nx3_mb = 0;
    std::vector<long long> block_order;
    ddc_transport::HostBuffer<float> rho;
    ddc_transport::HostBuffer<float> uu;
    ddc_transport::HostBuffer<float> uvec;
    ddc_transport::HostBuffer<float> bvec;
};

// Keep a small number of released host buffers for the reader thread.
class KHARMAHostBufferPool {
public:
    KHARMAStagedDump take() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffers_.empty()) return {};
        auto value=std::move(buffers_.back()); buffers_.pop_back(); return value;
    }
    void recycle(KHARMAStagedDump&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffers_.size()<2) buffers_.push_back(std::move(value));
    }
private:
    std::mutex mutex_;
    std::vector<KHARMAStagedDump> buffers_;
};

// Fixed snapshot slots and a small pinned upload pool. prepare() is ordered on
// an independent execution instance; fence() makes returned models readable.
// Slot reuse is legal only after its previous ray kernel has completed.
class KHARMAUploadWorkspace {
public:
    KHARMAUploadWorkspace(const GRMHDRadiationModel<DefaultReal>& first, size_t slots);
    ~KHARMAUploadWorkspace();
    KHARMAUploadWorkspace(const KHARMAUploadWorkspace&) = delete;
    KHARMAUploadWorkspace& operator=(const KHARMAUploadWorkspace&) = delete;
    GRMHDRadiationModel<DefaultReal> prepare(const KHARMAStagedDump&, const KHARMALoadOptions&, size_t slot);
    void fence();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

GRMHDRadiationModel<DefaultReal> load_kharma_model_from_phdf(const KHARMALoadOptions& opt);
KHARMAStagedDump read_kharma_staged_dump(const KHARMALoadOptions& opt);
KHARMAStagedDump read_kharma_staged_dump_resolved(const KHARMALoadOptions& opt, KHARMAStagedDump reuse = {});
GRMHDRadiationModel<DefaultReal> materialize_kharma_model_from_staged(const KHARMAStagedDump& staged,
                                                                       const KHARMALoadOptions& opt);
DefaultReal read_kharma_dump_time(
    const std::string& dump_path,
    const DDCInputOptions& ddc = DDCInputOptions::from_environment());

} // namespace kpolaris
