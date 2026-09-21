#pragma once

// Runtime-internal HDF5 adapter.  This declaration is paired with a compiled
// translation unit and is intentionally excluded from the installed
// header-only KPolaris::kpolaris API.  Installed command-line runtimes still
// contain and use the loader.

#include <cstddef>
#include <string>
#include <vector>

#include "geometry/binary_trajectory.hpp"

namespace kpolaris {

inline constexpr const char* binary_trajectory_schema_name =
    "kpolaris.binary_trajectory.v1";
inline constexpr int binary_trajectory_schema_version = 1;
inline constexpr const char* paper_trajectory_generator_version = "1.3";
inline constexpr const char* paper_merger_reach_contract =
    "selected_merger_state_fixed_tolerance_v1";

enum class BinaryTrajectoryHdf5Format : int {
    automatic = 0,
    native = 1,
    combi_ressler = 2
};

struct BinaryTrajectoryLoadOptions {
    BinaryTrajectoryHdf5Format format =
        BinaryTrajectoryHdf5Format::automatic;

    // When enabled, a verified W=1 endpoint is continued as one uniformly
    // moving Kerr remnant.  All other table queries remain strict-bound.
    bool enable_future_postmerger_extension = true;

    // The published Combi--Ressler postprocessor has an unequal-mass branch
    // that puts all remnant mass in one SKS term.  Repairing it to Mf/2 in each
    // coincident term is required for the two terms to sum to one Kerr metric.
    bool repair_combi_ressler_merger = true;

    // Multiplies every geometric time, length, mass, and Kerr-a value while
    // leaving dimensionless velocities unchanged.  For example, use
    // 1/(m1+m2) to convert a legacy Cactus-unit table to M_initial=1.
    DefaultReal geometric_unit_scale = DefaultReal(1);

    // Relative comparison tolerance used only during host validation and
    // exact-remnant canonicalization.  Runtime interpolation follows the
    // file's declared linear or coupled cubic-Hermite policy, with no
    // tolerance-based endpoint clamping.
    DefaultReal relative_tolerance = sizeof(DefaultReal) <= sizeof(float) ?
        DefaultReal(2e-5) : DefaultReal(2e-10);

    // Optional provenance contract.  The generic ``table`` path leaves
    // these empty, while the paper selector fills every field so that a
    // legacy or caller-labelled table cannot silently masquerade as the
    // verified Combi--Ressler trajectory model.
    bool require_source_verified = false;
    std::string required_generator;
    std::string required_generator_version;
    std::string required_merger_reach_contract;
    std::string required_trajectory_model;
    std::string required_pn_terms;
    std::string required_source_doi;
    std::string required_source_verification;
    std::string required_upstream_cbwaves_sha256;
    std::string required_patched_cbwaves_sha256;
    bool require_merger_separation_reached = false;
};

struct BinaryTrajectoryMetadata {
    BinaryTrajectoryHdf5Format format =
        BinaryTrajectoryHdf5Format::automatic;
    std::string source_path;
    std::string schema;
    std::string units;
    std::string position_gauge;
    std::string spin_convention;
    std::string interpolation;
    std::string generator;
    std::string generator_version;
    std::string merger_reach_contract;
    std::string trajectory_model;
    std::string pn_terms;
    // -1 means that a generic/legacy file made no verification assertion.
    int source_verified = -1;
    std::string source_doi;
    std::string pn_4pn_scope;
    std::string source_verification;
    std::string upstream_cbwaves_sha256;
    std::string patched_cbwaves_sha256;
    int merger_separation_reached = -1;
    std::string trajectory_status;
    // -1 means the legacy/older file did not declare this semantic flag.
    int worldline_velocity_consistent = -1;
    std::string boost_velocity_model;
    size_t sample_count = 0;
    DefaultReal time_min = DefaultReal(0);
    DefaultReal time_max = DefaultReal(0);
    DefaultReal transition_start_time = DefaultReal(0);
    DefaultReal transition_end_time = DefaultReal(0);
    DefaultReal postmerger_start_time = DefaultReal(0);
    bool has_declared_transition_endpoints = false;
    bool has_merger_transition = false;
    bool has_exact_postmerger_tail = false;
    bool future_postmerger_extension_enabled = false;
    bool repaired_legacy_merger = false;
    std::vector<std::string> warnings;
};

struct LoadedBinaryTrajectory {
    BinaryTrajectoryProvider<DefaultReal> provider;
    BinaryTrajectoryMetadata metadata;
};

// Native schema (all datasets are IEEE numeric arrays):
//   /trajectory/t              [N]
//   /trajectory/mass           [N,2]
//   /trajectory/position       [N,2,3]
//   /trajectory/velocity       [N,2,3]
//   /trajectory/kerr_a         [N,2,3]
//   /trajectory/merger_weight  [N]
// Required /trajectory string attributes are schema, units, position_gauge,
// spin_convention="kerr_a", interpolation ("linear" or
// "cubic_hermite_position_velocity"), generator, trajectory_model, pn_terms,
// and boost_velocity_model; schema_version and
// worldline_velocity_consistent are integer attributes.
// Cubic Hermite additionally requires worldline_velocity_consistent=1 and
// boost_velocity_model="derivative_of_position".  Only position/velocity are
// coupled C1; mass, Kerr-a and merger_weight remain piecewise linear.  Values
// must already share one geometric unit system.
//
// The legacy importer accepts the flat datasets written by the public
// Combi--Ressler trajectory_tools.py: t, x1...z2, vx1...vz2, either
// m1_full/m2_full or scalar m1/m2, and either a1x...a2z or dimensionless
// s1x...s2z.  Partial alternatives are rejected rather than guessed.
LoadedBinaryTrajectory load_binary_trajectory_hdf5(
    const std::string& path,
    const BinaryTrajectoryLoadOptions& options = BinaryTrajectoryLoadOptions());

} // namespace kpolaris
