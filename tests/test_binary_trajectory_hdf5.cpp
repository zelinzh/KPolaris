#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <H5Cpp.h>
#include <Kokkos_Core.hpp>
#include <unistd.h>

#include "geometry/binary_trajectory_hdf5.hpp"

namespace {

using Real = kpolaris::DefaultReal;

struct TemporaryHdf5 {
    explicit TemporaryHdf5(const std::string& label)
        : path("/tmp/kpolaris_" + label + "_" +
               std::to_string(static_cast<long long>(::getpid())) + ".h5") {
        std::remove(path.c_str());
    }
    ~TemporaryHdf5() { std::remove(path.c_str()); }
    std::string path;
};

void require_true(const std::string& name, bool condition) {
    if (!condition) throw std::runtime_error(name);
}

void require_close(const std::string& name, Real got, Real expected,
                   Real tolerance = sizeof(Real) <= sizeof(float) ?
                       Real(2e-5) : Real(2e-12)) {
    const Real scale = std::max<Real>(Real(1), std::abs(expected));
    if (!std::isfinite(static_cast<double>(got)) ||
        std::abs(got - expected) > tolerance * scale) {
        throw std::runtime_error(name + " mismatch: got " +
                                 std::to_string(got) + ", expected " +
                                 std::to_string(expected));
    }
}

void write_string_attribute(H5::H5Object& object, const std::string& name,
                            const std::string& value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute attribute = object.createAttribute(name, type, space);
    const char* pointer = value.c_str();
    attribute.write(type, &pointer);
}

void write_integer_attribute(H5::H5Object& object, const std::string& name,
                             int value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::Attribute attribute = object.createAttribute(
        name, H5::PredType::NATIVE_INT, space);
    attribute.write(H5::PredType::NATIVE_INT, &value);
}

void write_dataset(H5::Group& group, const std::string& name,
                   const std::vector<hsize_t>& dimensions,
                   const std::vector<double>& data) {
    size_t expected = 1;
    for (hsize_t dimension : dimensions) {
        expected *= static_cast<size_t>(dimension);
    }
    if (expected != data.size()) {
        throw std::runtime_error("bad test fixture size for " + name);
    }
    H5::DataSpace space(static_cast<int>(dimensions.size()),
                        dimensions.data());
    H5::DataSet dataset = group.createDataSet(
        name, H5::PredType::NATIVE_DOUBLE, space);
    dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}

void write_scalar_dataset(H5::Group& group, const std::string& name,
                          double value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::DataSet dataset = group.createDataSet(
        name, H5::PredType::NATIVE_DOUBLE, space);
    dataset.write(&value, H5::PredType::NATIVE_DOUBLE);
}

void write_native_fixture(const std::string& path,
                          bool nonmonotonic_time = false,
                          bool broken_remnant = false,
                          bool bad_spin_chi = false,
                          bool verified_paper = false,
                          bool paper_merger_reached = true,
                          bool current_completion_contract = true) {
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("/trajectory");
    write_string_attribute(group, "schema",
                           kpolaris::binary_trajectory_schema_name);
    write_integer_attribute(group, "schema_version",
                            kpolaris::binary_trajectory_schema_version);
    write_string_attribute(group, "units", "G=c=M_ref=1");
    write_string_attribute(group, "position_gauge", "harmonic");
    write_string_attribute(group, "spin_convention", "kerr_a");
    write_string_attribute(group, "interpolation", "linear");
    write_integer_attribute(group, "worldline_velocity_consistent", 0);
    write_string_attribute(group, "boost_velocity_model",
                           "paper_eq16_independent_blend");
    write_string_attribute(group, "generator", verified_paper ?
        "scripts/generate_paper_sks_trajectory.py" : "unit-test");
    write_string_attribute(group, "trajectory_model", verified_paper ?
        "paper_cbwaves_4pn_local" : "4PN test table");
    write_string_attribute(group, "pn_terms", verified_paper ?
        "PN,2PN,SO,SS,RR,PNSO,3PN,1RR,2PNSO,RRSO,RRSS,4PN" :
        "N,1PN,2PN,3PN,4PN,SO,SS,RR");
    if (verified_paper) {
        if (current_completion_contract) {
            write_string_attribute(
                group, "generator_version",
                kpolaris::paper_trajectory_generator_version);
            write_string_attribute(
                group, "merger_reach_contract",
                kpolaris::paper_merger_reach_contract);
        }
        write_integer_attribute(group, "merger_separation_reached",
                                paper_merger_reached ? 1 : 0);
        write_string_attribute(group, "trajectory_status",
                               paper_merger_reached ?
                                   "merger_separation_reached" :
                                   "diagnostic_truncated");
        H5::Group root = file.openGroup("/");
        write_integer_attribute(root, "source_verified", 1);
        write_string_attribute(root, "source_doi",
                               "10.5281/zenodo.10841021");
        H5::Group provenance = file.createGroup("/provenance");
        write_string_attribute(provenance, "source_verification",
                               "verified-zenodo-record-10841021");
        write_string_attribute(
            provenance, "upstream_cbwaves_sha256",
            "4a094a9bbac3b2bc142015b70afd5dc939898b7b38a34b05917dd94e94829a9a");
        write_string_attribute(
            provenance, "patched_cbwaves_sha256",
            "0a28a193f0964164876968b159b9df0c261190f4a605de138592878e967c6067");
        if (current_completion_contract) {
            H5::Group remnant = file.createGroup("/remnant");
            write_scalar_dataset(remnant, "transition_start", 0.0);
            write_scalar_dataset(remnant, "transition_end", 2.0);
        }
    }

    write_dataset(group, "t", {4},
                  nonmonotonic_time ?
                      std::vector<double>{0, 1, 1, 3} :
                      std::vector<double>{0, 1, 2, 3});
    write_dataset(group, "mass", {4, 2}, {
        0.60, 0.40,
        0.55, 0.425,
        0.475, 0.475,
        0.475, 0.475});
    write_dataset(group, "position", {4, 2, 3}, {
         2.0,  0.0, 0.0,  -3.0,  0.0, 0.0,
         0.5,  0.5, 0.0,  -0.4, -0.2, 0.0,
         1.0,  2.0, 0.0,   1.0,  2.0, 0.0,
         1.1,  2.0, 0.0,   broken_remnant ? 1.4 : 1.1, 2.0, 0.0});
    write_dataset(group, "velocity", {4, 2, 3}, {
        0.0,  0.20, 0.0,  0.0, -0.30, 0.0,
        0.05, 0.10, 0.0,  0.02, -0.08, 0.0,
        0.10, 0.00, 0.0,  0.10,  0.00, 0.0,
        0.10, 0.00, 0.0,  0.10,  0.00, 0.0});
    write_dataset(group, "kerr_a", {4, 2, 3}, {
        0.0, 0.0,  0.12,  0.0, 0.0, -0.08,
        0.0, 0.0,  0.35,  0.0, 0.0,  0.30,
        0.0, 0.0,  0.65,  0.0, 0.0,  0.65,
        0.0, 0.0,  0.65,  0.0, 0.0,  0.65});
    write_dataset(group, "spin_chi", {4, 2, 3}, {
        0.0, 0.0, bad_spin_chi ? 0.25 : 0.20,
            0.0, 0.0, -0.20,
        0.0, 0.0, 0.35 / 0.55,
            0.0, 0.0, 0.30 / 0.425,
        0.0, 0.0, 0.65 / 0.475,
            0.0, 0.0, 0.65 / 0.475,
        0.0, 0.0, 0.65 / 0.475,
            0.0, 0.0, 0.65 / 0.475});
    const double near_one = sizeof(Real) <= sizeof(float) ?
        static_cast<double>(std::nextafter(Real(1), Real(0))) :
        std::nextafter(1.0, 0.0);
    write_dataset(group, "merger_weight", {4},
                  {0.0, near_one, 1.0, 1.0});
}

void write_hermite_fixture(const std::string& path,
                           bool overspeed = false,
                           bool include_semantics = true) {
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("/trajectory");
    write_string_attribute(group, "schema",
                           kpolaris::binary_trajectory_schema_name);
    write_integer_attribute(group, "schema_version",
                            kpolaris::binary_trajectory_schema_version);
    write_string_attribute(group, "units", "G=c=M_ref=1");
    write_string_attribute(group, "position_gauge", "harmonic");
    write_string_attribute(group, "spin_convention", "kerr_a");
    write_string_attribute(group, "interpolation",
                           "cubic_hermite_position_velocity");
    write_string_attribute(group, "generator", "unit-test");
    write_string_attribute(group, "trajectory_model", "Hermite test table");
    write_string_attribute(group, "pn_terms", "test-only");
    if (include_semantics) {
        write_integer_attribute(group, "worldline_velocity_consistent", 1);
        write_string_attribute(group, "boost_velocity_model",
                               "derivative_of_position");
    }
    write_dataset(group, "t", {2}, {0.0, 1.0});
    write_dataset(group, "mass", {2, 2},
                  {0.6, 0.4, 0.6, 0.4});
    write_dataset(group, "position", {2, 2, 3}, overspeed ?
        std::vector<double>{
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0} :
        std::vector<double>{
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0, -0.25, 0.0, 0.0});
    write_dataset(group, "velocity", {2, 2, 3}, overspeed ?
        std::vector<double>{
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0} :
        std::vector<double>{
            0.2, 0.0, 0.0, -0.1, 0.0, 0.0,
            0.8, 0.0, 0.0, -0.4, 0.0, 0.0});
    write_dataset(group, "kerr_a", {2, 2, 3}, {
        0.0, 0.0, 0.1, 0.0, 0.0, -0.05,
        0.0, 0.0, 0.1, 0.0, 0.0, -0.05});
    write_dataset(group, "merger_weight", {2}, {0.0, 0.0});
}

void write_legacy_merger_fixture(const std::string& path) {
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    const std::vector<double> time = {0, 1, 2, 3};
    write_dataset(root, "t", {4}, time);
    write_dataset(root, "x1", {4}, {2.0, 0.5, 1.0, 1.1});
    write_dataset(root, "y1", {4}, {0.0, 0.5, 2.0, 2.0});
    write_dataset(root, "z1", {4}, {0.0, 0.0, 0.0, 0.0});
    write_dataset(root, "x2", {4}, {-3.0, -0.5, 1.0, 1.1});
    write_dataset(root, "y2", {4}, {0.0, -0.4, 2.0, 2.0});
    write_dataset(root, "z2", {4}, {0.0, 0.0, 0.0, 0.0});
    write_dataset(root, "vx1", {4}, {0.0, 0.05, 0.1, 0.1});
    write_dataset(root, "vy1", {4}, {0.2, 0.1, 0.0, 0.0});
    write_dataset(root, "vz1", {4}, {0.0, 0.0, 0.0, 0.0});
    write_dataset(root, "vx2", {4}, {0.0, 0.02, 0.1, 0.1});
    write_dataset(root, "vy2", {4}, {-0.3, -0.08, 0.0, 0.0});
    write_dataset(root, "vz2", {4}, {0.0, 0.0, 0.0, 0.0});

    // Reproduce the published unequal-mass postprocessor branch: all Mf is
    // assigned to term 1 and term 2 tends to zero.  The importer must repair
    // this to Mf/2 + Mf/2 using the inferred W={0,.5,1,1}.
    write_dataset(root, "m1_full", {4}, {0.60, 0.775, 0.95, 0.95});
    write_dataset(root, "m2_full", {4}, {0.40, 0.20, 0.00, 0.00});
    write_scalar_dataset(root, "m1", 0.60);
    write_scalar_dataset(root, "m2", 0.40);

    write_dataset(root, "a1x", {4}, {0, 0, 0, 0});
    write_dataset(root, "a1y", {4}, {0, 0, 0, 0});
    write_dataset(root, "a1z", {4}, {0.12, 0.35, 0.65, 0.65});
    write_dataset(root, "a2x", {4}, {0, 0, 0, 0});
    write_dataset(root, "a2y", {4}, {0, 0, 0, 0});
    write_dataset(root, "a2z", {4}, {-0.08, 0.30, 0.65, 0.65});
    write_scalar_dataset(root, "t_postmerger", 2.0);
    write_scalar_dataset(root, "a_x_remnant", 0.0);
    write_scalar_dataset(root, "a_y_remnant", 0.0);
    write_scalar_dataset(root, "a_z_remnant", 0.65);
}

void write_legacy_spin_fixture(const std::string& path) {
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    write_dataset(root, "t", {2}, {0, 2});
    write_dataset(root, "x1", {2}, {2, 1.8});
    write_dataset(root, "y1", {2}, {0, 0.4});
    write_dataset(root, "z1", {2}, {0, 0});
    write_dataset(root, "x2", {2}, {-3, -2.7});
    write_dataset(root, "y2", {2}, {0, -0.6});
    write_dataset(root, "z2", {2}, {0, 0});
    write_dataset(root, "vx1", {2}, {0, -0.1});
    write_dataset(root, "vy1", {2}, {0.2, 0.2});
    write_dataset(root, "vz1", {2}, {0, 0});
    write_dataset(root, "vx2", {2}, {0, 0.15});
    write_dataset(root, "vy2", {2}, {-0.3, -0.3});
    write_dataset(root, "vz2", {2}, {0, 0});
    write_scalar_dataset(root, "m1", 0.60);
    write_scalar_dataset(root, "m2", 0.40);
    write_dataset(root, "s1x", {2}, {0, 0});
    write_dataset(root, "s1y", {2}, {0, 0});
    write_dataset(root, "s1z", {2}, {0.2, 0.2});
    write_dataset(root, "s2x", {2}, {0, 0});
    write_dataset(root, "s2y", {2}, {0, 0});
    write_dataset(root, "s2z", {2}, {-0.2, -0.2});
}

void test_native_interpolation_bounds_and_device() {
    TemporaryHdf5 fixture("native_binary_trajectory");
    write_native_fixture(fixture.path);
    const auto loaded = kpolaris::load_binary_trajectory_hdf5(fixture.path);
    require_true("native format detection",
        loaded.metadata.format ==
            kpolaris::BinaryTrajectoryHdf5Format::native);
    require_true("native exact remnant detection",
                 loaded.metadata.has_exact_postmerger_tail);
    require_true("native metadata provenance",
                 loaded.metadata.trajectory_model == "4PN test table");

    const auto midpoint = loaded.provider.state(Real(0.5));
    require_true("native midpoint validity",
                 midpoint.valid && midpoint.in_table_bounds &&
                 !midpoint.future_extended && !midpoint.postmerger);
    const Real near_one = std::nextafter(Real(1), Real(0));
    require_close("native linear W", midpoint.merger_weight,
                  near_one * Real(0.5));
    require_close("native linear mass1", midpoint.mass1, Real(0.575));
    require_close("native linear position1.x", midpoint.position1.x,
                  Real(1.25));
    require_close("native linear kerr a1.z", midpoint.kerr_a1.z,
                  Real(0.235));
    const auto near_endpoint = loaded.provider.state(Real(1));
    require_true("near-one smooth-window sample is not postmerger",
                 near_endpoint.valid && !near_endpoint.postmerger &&
                 near_endpoint.merger_weight < Real(1));

    const auto before = loaded.provider.state(Real(-0.01));
    require_true("native lower bound is strict", !before.valid);
    require_close("lower-bound RK guard mass is finite", before.mass1,
                  Real(0.60));
    require_close("lower-bound RK guard position is finite",
                  before.position1.x, Real(2));
    require_true("lower-bound query excluded by domain",
                 !loaded.provider.has_state(Real(-0.01)));
    const auto nan_state = loaded.provider.state(
        std::numeric_limits<Real>::quiet_NaN());
    require_true("NaN trajectory query is zero-invalid",
                 !nan_state.valid && nan_state.mass1 == Real(0) &&
                 nan_state.position1.x == Real(0));
    require_true("infinite trajectory query excluded by domain",
                 !loaded.provider.has_state(
                     std::numeric_limits<Real>::infinity()));
    const auto future = loaded.provider.state(Real(4));
    require_true("native physical remnant extension",
                 future.valid && !future.in_table_bounds &&
                 future.future_extended && future.postmerger);
    require_close("future remnant position1", future.position1.x, Real(1.2));
    require_close("future remnant position2", future.position2.x, Real(1.2));
    require_close("future remnant half mass", future.mass1, Real(0.475));
    require_close("future remnant duplicate a", future.kerr_a1.z,
                  future.kerr_a2.z);

    kpolaris::BinaryTrajectoryLoadOptions strict_options;
    strict_options.enable_future_postmerger_extension = false;
    const auto strict_loaded = kpolaris::load_binary_trajectory_hdf5(
        fixture.path, strict_options);
    require_true("disabled future extension is strict",
                 !strict_loaded.provider.state(Real(4)).valid);

    Kokkos::View<Real*> device_result("trajectory_device_result", 6);
    const auto provider = loaded.provider;
    Kokkos::parallel_for("binary_trajectory_device_smoke", 1,
        KOKKOS_LAMBDA(const int) {
            const auto interpolated = provider.state(Real(0.5));
            const auto extended = provider.state(Real(4));
            device_result(0) = interpolated.mass1;
            device_result(1) = interpolated.position1.x;
            device_result(2) = interpolated.valid;
            device_result(3) = extended.position1.x;
            device_result(4) = extended.future_extended;
            device_result(5) = extended.separation;
        });
    const auto host_result = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), device_result);
    require_close("device interpolation mass", host_result(0), Real(0.575));
    require_close("device interpolation position", host_result(1), Real(1.25));
    require_close("device interpolation validity", host_result(2), Real(1));
    require_close("device future position", host_result(3), Real(1.2));
    require_close("device future extension flag", host_result(4), Real(1));
    require_close("device exact remnant separation", host_result(5), Real(0));
}

void test_legacy_merger_repair() {
    TemporaryHdf5 fixture("legacy_binary_trajectory_merger");
    write_legacy_merger_fixture(fixture.path);
    const auto loaded = kpolaris::load_binary_trajectory_hdf5(fixture.path);
    require_true("legacy format detection",
        loaded.metadata.format ==
            kpolaris::BinaryTrajectoryHdf5Format::combi_ressler);
    require_true("legacy repair recorded",
                 loaded.metadata.repaired_legacy_merger);
    require_true("legacy exact tail after repair",
                 loaded.metadata.has_exact_postmerger_tail);

    const auto transition = loaded.provider.state(Real(1));
    require_close("legacy inferred W", transition.merger_weight, Real(0.5));
    require_close("legacy repaired transition m1", transition.mass1,
                  Real(0.5375));
    require_close("legacy repaired transition m2", transition.mass2,
                  Real(0.4375));
    const auto remnant = loaded.provider.state(Real(2));
    require_true("legacy remnant flag", remnant.postmerger != 0);
    require_close("legacy remnant m1", remnant.mass1, Real(0.475));
    require_close("legacy remnant m2", remnant.mass2, Real(0.475));
    require_close("legacy remnant separation", remnant.separation, Real(0));
    require_close("legacy full a per half term", remnant.kerr_a1.z,
                  Real(0.65));
    const auto future = loaded.provider.state(Real(4));
    require_true("legacy remnant future extension is disabled",
                 !future.valid &&
                 !loaded.metadata.future_postmerger_extension_enabled);
    require_true("legacy extension warning recorded",
                 !loaded.metadata.warnings.empty());

    kpolaris::BinaryTrajectoryLoadOptions no_repair;
    no_repair.repair_combi_ressler_merger = false;
    bool rejected = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(fixture.path, no_repair);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require_true("uncorrected unequal-mass legacy tail rejected", rejected);
}

void test_coupled_hermite_semantics_and_speed_validation() {
    TemporaryHdf5 fixture("native_binary_trajectory_hermite");
    write_hermite_fixture(fixture.path);
    const auto loaded = kpolaris::load_binary_trajectory_hdf5(fixture.path);
    require_true("Hermite interpolation mode loaded",
        loaded.provider.interpolation == static_cast<int>(
            kpolaris::BinaryTrajectoryInterpolation::
                cubic_hermite_position_velocity));
    require_true("Hermite semantic flag retained",
                 loaded.metadata.worldline_velocity_consistent == 1);
    require_true("Hermite boost semantic retained",
                 loaded.metadata.boost_velocity_model ==
                     "derivative_of_position");
    const auto middle = loaded.provider.state(Real(0.5));
    require_close("Hermite position1", middle.position1.x, Real(0.175));
    require_close("Hermite derivative velocity1", middle.velocity1.x,
                  Real(0.5));
    require_close("Hermite position2", middle.position2.x, Real(-0.0875));
    require_close("Hermite derivative velocity2", middle.velocity2.x,
                  Real(-0.25));

    Kokkos::View<Real*> device_result("trajectory_hermite_device_result", 2);
    const auto provider = loaded.provider;
    Kokkos::parallel_for("binary_trajectory_hermite_device", 1,
        KOKKOS_LAMBDA(const int) {
            const auto state = provider.state(Real(0.5));
            device_result(0) = state.position1.x;
            device_result(1) = state.velocity1.x;
        });
    const auto host_result = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), device_result);
    require_close("device Hermite position", host_result(0), Real(0.175));
    require_close("device Hermite derivative velocity", host_result(1),
                  Real(0.5));

    TemporaryHdf5 missing_semantics("native_hermite_missing_semantics");
    write_hermite_fixture(missing_semantics.path, false, false);
    bool rejected_semantics = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(
            missing_semantics.path);
    } catch (const std::runtime_error&) {
        rejected_semantics = true;
    }
    require_true("Hermite table without velocity semantics rejected",
                 rejected_semantics);

    TemporaryHdf5 overspeed("native_hermite_overspeed");
    write_hermite_fixture(overspeed.path, true, true);
    bool rejected_speed = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(overspeed.path);
    } catch (const std::runtime_error&) {
        rejected_speed = true;
    }
    require_true("between-knot superluminal Hermite boost rejected",
                 rejected_speed);
}

void test_legacy_dimensionless_spin_and_unit_scale() {
    TemporaryHdf5 fixture("legacy_binary_trajectory_spin");
    write_legacy_spin_fixture(fixture.path);
    kpolaris::BinaryTrajectoryLoadOptions options;
    options.geometric_unit_scale = Real(0.5);
    const auto loaded = kpolaris::load_binary_trajectory_hdf5(
        fixture.path, options);
    const auto state = loaded.provider.state(Real(0));
    require_close("scaled legacy mass1", state.mass1, Real(0.3));
    require_close("scaled legacy position1", state.position1.x, Real(1));
    require_close("scaled legacy time max", loaded.metadata.time_max, Real(1));
    require_close("legacy chi to scaled Kerr a1", state.kerr_a1.z,
                  Real(0.06));
    require_close("legacy chi to scaled Kerr a2", state.kerr_a2.z,
                  Real(-0.04));
    require_true("inspiral-only legacy table has strict upper bound",
                 !loaded.provider.state(Real(1.1)).valid);
}

void test_malformed_native_rejected() {
    TemporaryHdf5 missing_linear_semantics(
        "bad_binary_trajectory_missing_linear_semantics");
    write_native_fixture(missing_linear_semantics.path);
    {
        H5::H5File file(missing_linear_semantics.path, H5F_ACC_RDWR);
        H5::Group group = file.openGroup("/trajectory");
        H5Adelete(group.getId(), "worldline_velocity_consistent");
        H5Adelete(group.getId(), "boost_velocity_model");
    }
    bool rejected_linear_semantics = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(
            missing_linear_semantics.path);
    } catch (const std::runtime_error&) {
        rejected_linear_semantics = true;
    }
    require_true("linear table without velocity semantics rejected",
                 rejected_linear_semantics);

    TemporaryHdf5 nonmonotonic("bad_binary_trajectory_time");
    write_native_fixture(nonmonotonic.path, true, false);
    bool rejected_time = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(nonmonotonic.path);
    } catch (const std::runtime_error&) {
        rejected_time = true;
    }
    require_true("nonmonotonic native time rejected", rejected_time);

    TemporaryHdf5 broken_remnant("bad_binary_trajectory_remnant");
    write_native_fixture(broken_remnant.path, false, true);
    bool rejected_remnant = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(broken_remnant.path);
    } catch (const std::runtime_error&) {
        rejected_remnant = true;
    }
    require_true("noncoincident native remnant rejected", rejected_remnant);

    TemporaryHdf5 bad_chi("bad_binary_trajectory_spin_chi");
    write_native_fixture(bad_chi.path, false, false, true);
    bool rejected_chi = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(bad_chi.path);
    } catch (const std::runtime_error&) {
        rejected_chi = true;
    }
    require_true("inconsistent optional spin_chi rejected", rejected_chi);
}

void test_verified_paper_contract() {
    TemporaryHdf5 verified("verified_paper_binary_trajectory");
    write_native_fixture(verified.path, false, false, false, true);
    kpolaris::BinaryTrajectoryLoadOptions contract;
    contract.format = kpolaris::BinaryTrajectoryHdf5Format::native;
    contract.require_source_verified = true;
    contract.require_merger_separation_reached = true;
    contract.required_generator =
        "scripts/generate_paper_sks_trajectory.py";
    contract.required_generator_version =
        kpolaris::paper_trajectory_generator_version;
    contract.required_merger_reach_contract =
        kpolaris::paper_merger_reach_contract;
    contract.required_trajectory_model = "paper_cbwaves_4pn_local";
    contract.required_pn_terms =
        "PN,2PN,SO,SS,RR,PNSO,3PN,1RR,2PNSO,RRSO,RRSS,4PN";
    contract.required_source_doi = "10.5281/zenodo.10841021";
    contract.required_source_verification =
        "verified-zenodo-record-10841021";
    contract.required_upstream_cbwaves_sha256 =
        "4a094a9bbac3b2bc142015b70afd5dc939898b7b38a34b05917dd94e94829a9a";
    contract.required_patched_cbwaves_sha256 =
        "0a28a193f0964164876968b159b9df0c261190f4a605de138592878e967c6067";
    const auto loaded = kpolaris::load_binary_trajectory_hdf5(
        verified.path, contract);
    require_true("verified paper source flag retained",
                 loaded.metadata.source_verified == 1);
    require_true("verified paper DOI retained",
                 loaded.metadata.source_doi ==
                     "10.5281/zenodo.10841021");
    require_true("declared transition endpoints retained",
                 loaded.metadata.has_declared_transition_endpoints);
    require_close("exact declared transition start",
                  loaded.metadata.transition_start_time, Real(0));
    require_close("exact declared transition end",
                  loaded.metadata.transition_end_time, Real(2));

    TemporaryHdf5 generic("generic_not_verified_paper");
    write_native_fixture(generic.path);
    bool rejected = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(generic.path, contract);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require_true("generic table cannot claim verified paper selector",
                 rejected);

    TemporaryHdf5 truncated("verified_but_truncated_paper");
    write_native_fixture(truncated.path, false, false, false, true, false);
    bool rejected_truncated = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(truncated.path,
                                                    contract);
    } catch (const std::runtime_error&) {
        rejected_truncated = true;
    }
    require_true("verified but truncated paper trajectory rejected",
                 rejected_truncated);

    TemporaryHdf5 stale_contract("stale_paper_completion_contract");
    write_native_fixture(stale_contract.path, false, false, false, true,
                         true, false);
    bool rejected_stale_contract = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(stale_contract.path,
                                                    contract);
    } catch (const std::runtime_error&) {
        rejected_stale_contract = true;
    }
    require_true("paper trajectory without fixed completion contract rejected",
                 rejected_stale_contract);

    TemporaryHdf5 bad_endpoint("bad_declared_transition_endpoint");
    write_native_fixture(bad_endpoint.path, false, false, false, true);
    {
        H5::H5File file(bad_endpoint.path, H5F_ACC_RDWR);
        H5::DataSet dataset = file.openDataSet("/remnant/transition_start");
        // This lies inside the general field-validation tolerance but is not
        // literally a stored time sample.  The endpoint contract is exact.
        const double non_sample_endpoint = 1.0e-11;
        dataset.write(&non_sample_endpoint, H5::PredType::NATIVE_DOUBLE);
    }
    bool rejected_bad_endpoint = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(bad_endpoint.path,
                                                    contract);
    } catch (const std::runtime_error&) {
        rejected_bad_endpoint = true;
    }
    require_true("non-sample transition endpoint rejected",
                 rejected_bad_endpoint);

    TemporaryHdf5 missing_transition(
        "declared_endpoints_without_merger_transition");
    write_native_fixture(missing_transition.path, false, false, false, true);
    {
        H5::H5File file(missing_transition.path, H5F_ACC_RDWR);
        H5::DataSet dataset =
            file.openDataSet("/trajectory/merger_weight");
        const double all_inspiral[4] = {0.0, 0.0, 0.0, 0.0};
        dataset.write(all_inspiral, H5::PredType::NATIVE_DOUBLE);
    }
    bool rejected_missing_transition = false;
    try {
        (void)kpolaris::load_binary_trajectory_hdf5(
            missing_transition.path, contract);
    } catch (const std::runtime_error&) {
        rejected_missing_transition = true;
    }
    require_true("declared endpoints without a merger transition rejected",
                 rejected_missing_transition);
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        test_native_interpolation_bounds_and_device();
        test_legacy_merger_repair();
        test_legacy_dimensionless_spin_and_unit_scale();
        test_malformed_native_rejected();
        test_coupled_hermite_semantics_and_speed_validation();
        test_verified_paper_contract();
    } catch (...) {
        Kokkos::finalize();
        throw;
    }
    Kokkos::finalize();
    std::cout << "KPolaris binary trajectory HDF5 tests passed\n";
    return 0;
}
