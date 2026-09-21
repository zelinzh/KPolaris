#include "geometry/binary_trajectory_hdf5.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <H5Cpp.h>
#include <Kokkos_Core.hpp>

namespace kpolaris {
namespace {

using Real = DefaultReal;
using FieldArray =
    std::array<std::vector<Real>, binary_trajectory_field_count>;

struct HostTrajectoryTable {
    std::vector<Real> time;
    FieldArray field;
};

struct NumericDataset {
    std::vector<double> values;
    std::vector<hsize_t> dimensions;
};

size_t field_index(BinaryTrajectoryField field) {
    return static_cast<size_t>(static_cast<int>(field));
}

std::vector<Real>& values(HostTrajectoryTable& table,
                          BinaryTrajectoryField field) {
    return table.field[field_index(field)];
}

const std::vector<Real>& values(const HostTrajectoryTable& table,
                                BinaryTrajectoryField field) {
    return table.field[field_index(field)];
}

bool link_exists(H5::H5File& file, const std::string& path) {
    return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

bool attribute_exists(H5::H5Object& object, const std::string& name) {
    return H5Aexists(object.getId(), name.c_str()) > 0;
}

std::string trim_copy(const std::string& input) {
    size_t first = 0;
    while (first < input.size() &&
           std::isspace(static_cast<unsigned char>(input[first]))) {
        ++first;
    }
    size_t last = input.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(input[last - 1]))) {
        --last;
    }
    return input.substr(first, last - first);
}

std::string lowercase_copy(std::string input) {
    for (char& c : input) {
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }
    return input;
}

std::string read_string_attribute(H5::H5Object& object,
                                  const std::string& name) {
    if (!attribute_exists(object, name)) {
        throw std::runtime_error("missing required HDF5 attribute '" + name +
                                 "'");
    }
    H5::Attribute attribute = object.openAttribute(name);
    H5::DataType type = attribute.getDataType();
    if (H5Tget_class(type.getId()) != H5T_STRING) {
        throw std::runtime_error("HDF5 attribute '" + name +
                                 "' must be a string");
    }
    if (H5Tis_variable_str(type.getId()) > 0) {
        std::string output;
        attribute.read(type, output);
        return trim_copy(output);
    }
    const size_t length = type.getSize();
    std::vector<char> buffer(length + 1, '\0');
    attribute.read(type, buffer.data());
    buffer[length] = '\0';
    return trim_copy(std::string(buffer.data()));
}

std::string read_optional_string_attribute(H5::H5Object& object,
                                           const std::string& name) {
    return attribute_exists(object, name) ?
        read_string_attribute(object, name) : std::string();
}

int read_integer_attribute(H5::H5Object& object,
                           const std::string& name) {
    if (!attribute_exists(object, name)) {
        throw std::runtime_error("missing required HDF5 attribute '" + name +
                                 "'");
    }
    H5::Attribute attribute = object.openAttribute(name);
    const hssize_t count = attribute.getSpace().getSimpleExtentNpoints();
    if (count != 1) {
        throw std::runtime_error("HDF5 attribute '" + name +
                                 "' must be scalar");
    }
    int output = 0;
    attribute.read(H5::PredType::NATIVE_INT, &output);
    return output;
}

int read_optional_integer_attribute(H5::H5Object& object,
                                    const std::string& name,
                                    int fallback = -1) {
    return attribute_exists(object, name) ?
        read_integer_attribute(object, name) : fallback;
}

NumericDataset read_numeric_dataset(H5::H5File& file,
                                    const std::string& path) {
    if (!link_exists(file, path)) {
        throw std::runtime_error("missing required HDF5 dataset '" + path +
                                 "'");
    }
    H5::DataSet dataset = file.openDataSet(path);
    H5::DataSpace space = dataset.getSpace();
    const int rank = space.getSimpleExtentNdims();
    if (rank < 0) {
        throw std::runtime_error("invalid HDF5 dataspace for '" + path + "'");
    }
    NumericDataset output;
    output.dimensions.resize(static_cast<size_t>(rank));
    if (rank > 0) {
        space.getSimpleExtentDims(output.dimensions.data());
    }
    size_t count = 1;
    for (hsize_t dimension : output.dimensions) {
        if (dimension == 0) {
            throw std::runtime_error("HDF5 dataset '" + path +
                                     "' may not be empty");
        }
        if (dimension > std::numeric_limits<size_t>::max() / count) {
            throw std::runtime_error("HDF5 dataset size overflows for '" +
                                     path + "'");
        }
        count *= static_cast<size_t>(dimension);
    }
    output.values.resize(count);
    dataset.read(output.values.data(), H5::PredType::NATIVE_DOUBLE);
    return output;
}

void require_dimensions(const NumericDataset& dataset,
                        const std::vector<hsize_t>& expected,
                        const std::string& path) {
    if (dataset.dimensions != expected) {
        std::ostringstream message;
        message << "HDF5 dataset '" << path << "' has dimensions [";
        for (size_t i = 0; i < dataset.dimensions.size(); ++i) {
            if (i) message << ',';
            message << dataset.dimensions[i];
        }
        message << "]; expected [";
        for (size_t i = 0; i < expected.size(); ++i) {
            if (i) message << ',';
            message << expected[i];
        }
        message << ']';
        throw std::runtime_error(message.str());
    }
}

std::vector<Real> read_vector(H5::H5File& file, const std::string& path,
                              size_t expected_count,
                              Real scale = Real(1)) {
    const NumericDataset dataset = read_numeric_dataset(file, path);
    require_dimensions(dataset,
                       {static_cast<hsize_t>(expected_count)}, path);
    std::vector<Real> output(expected_count);
    for (size_t i = 0; i < expected_count; ++i) {
        output[i] = static_cast<Real>(dataset.values[i]) * scale;
    }
    return output;
}

Real read_scalar(H5::H5File& file, const std::string& path,
                 Real scale = Real(1)) {
    const NumericDataset dataset = read_numeric_dataset(file, path);
    if (dataset.values.size() != 1) {
        throw std::runtime_error("HDF5 dataset '" + path +
                                 "' must contain exactly one value");
    }
    return static_cast<Real>(dataset.values[0]) * scale;
}

bool close_scalar(Real left, Real right, Real tolerance) {
    const Real scale = std::max<Real>(
        Real(1), std::max(std::abs(left), std::abs(right)));
    return std::abs(left - right) <= tolerance * scale;
}

bool close_vector(const Vec3<Real>& left, const Vec3<Real>& right,
                  Real tolerance) {
    return close_scalar(left.x, right.x, tolerance) &&
           close_scalar(left.y, right.y, tolerance) &&
           close_scalar(left.z, right.z, tolerance);
}

Vec3<Real> vector_at(const HostTrajectoryTable& table,
                     BinaryTrajectoryField x_field,
                     BinaryTrajectoryField y_field,
                     BinaryTrajectoryField z_field,
                     size_t sample) {
    return Vec3<Real>(values(table, x_field)[sample],
                      values(table, y_field)[sample],
                      values(table, z_field)[sample]);
}

void set_vector_at(HostTrajectoryTable& table,
                   BinaryTrajectoryField x_field,
                   BinaryTrajectoryField y_field,
                   BinaryTrajectoryField z_field,
                   size_t sample, const Vec3<Real>& vector) {
    values(table, x_field)[sample] = vector.x;
    values(table, y_field)[sample] = vector.y;
    values(table, z_field)[sample] = vector.z;
}

Vec3<Real> position_at(const HostTrajectoryTable& table, int hole,
                       size_t sample) {
    if (hole == 0) {
        return vector_at(table, BinaryTrajectoryField::position1_x,
                         BinaryTrajectoryField::position1_y,
                         BinaryTrajectoryField::position1_z, sample);
    }
    return vector_at(table, BinaryTrajectoryField::position2_x,
                     BinaryTrajectoryField::position2_y,
                     BinaryTrajectoryField::position2_z, sample);
}

Vec3<Real> velocity_at(const HostTrajectoryTable& table, int hole,
                       size_t sample) {
    if (hole == 0) {
        return vector_at(table, BinaryTrajectoryField::velocity1_x,
                         BinaryTrajectoryField::velocity1_y,
                         BinaryTrajectoryField::velocity1_z, sample);
    }
    return vector_at(table, BinaryTrajectoryField::velocity2_x,
                     BinaryTrajectoryField::velocity2_y,
                     BinaryTrajectoryField::velocity2_z, sample);
}

Vec3<Real> kerr_a_at(const HostTrajectoryTable& table, int hole,
                     size_t sample) {
    if (hole == 0) {
        return vector_at(table, BinaryTrajectoryField::kerr_a1_x,
                         BinaryTrajectoryField::kerr_a1_y,
                         BinaryTrajectoryField::kerr_a1_z, sample);
    }
    return vector_at(table, BinaryTrajectoryField::kerr_a2_x,
                     BinaryTrajectoryField::kerr_a2_y,
                     BinaryTrajectoryField::kerr_a2_z, sample);
}

void set_position_at(HostTrajectoryTable& table, int hole, size_t sample,
                     const Vec3<Real>& vector) {
    if (hole == 0) {
        set_vector_at(table, BinaryTrajectoryField::position1_x,
                      BinaryTrajectoryField::position1_y,
                      BinaryTrajectoryField::position1_z, sample, vector);
    } else {
        set_vector_at(table, BinaryTrajectoryField::position2_x,
                      BinaryTrajectoryField::position2_y,
                      BinaryTrajectoryField::position2_z, sample, vector);
    }
}

void set_velocity_at(HostTrajectoryTable& table, int hole, size_t sample,
                     const Vec3<Real>& vector) {
    if (hole == 0) {
        set_vector_at(table, BinaryTrajectoryField::velocity1_x,
                      BinaryTrajectoryField::velocity1_y,
                      BinaryTrajectoryField::velocity1_z, sample, vector);
    } else {
        set_vector_at(table, BinaryTrajectoryField::velocity2_x,
                      BinaryTrajectoryField::velocity2_y,
                      BinaryTrajectoryField::velocity2_z, sample, vector);
    }
}

void set_kerr_a_at(HostTrajectoryTable& table, int hole, size_t sample,
                   const Vec3<Real>& vector) {
    if (hole == 0) {
        set_vector_at(table, BinaryTrajectoryField::kerr_a1_x,
                      BinaryTrajectoryField::kerr_a1_y,
                      BinaryTrajectoryField::kerr_a1_z, sample, vector);
    } else {
        set_vector_at(table, BinaryTrajectoryField::kerr_a2_x,
                      BinaryTrajectoryField::kerr_a2_y,
                      BinaryTrajectoryField::kerr_a2_z, sample, vector);
    }
}

Vec3<Real> average_vector(const Vec3<Real>& left,
                          const Vec3<Real>& right) {
    return (left + right) * Real(0.5);
}

void allocate_fields(HostTrajectoryTable& table, size_t sample_count) {
    for (std::vector<Real>& field : table.field) {
        field.assign(sample_count, Real(0));
    }
}

void require_finite(Real value, const std::string& description) {
    if (!std::isfinite(static_cast<double>(value))) {
        throw std::runtime_error("non-finite binary trajectory value in " +
                                 description);
    }
}

Real maximum_valid_boost_speed2() {
    // Leave enough headroom that 1-v^2 and gamma are represented reliably on
    // the selected scalar type.  The device metric then uses the exact
    // validated velocity rather than silently clamping gamma inconsistently.
    return Real(1) - (sizeof(Real) <= sizeof(float) ? Real(1e-5) : Real(1e-12));
}

Real cubic_value(Real c3, Real c2, Real c1, Real c0, Real u) {
    return ((c3 * u + c2) * u + c1) * u + c0;
}

void validate_hermite_segment_speed(
    const HostTrajectoryTable& table, size_t lower, int hole) {
    const size_t upper = lower + 1;
    const Real dt = table.time[upper] - table.time[lower];
    const BinaryTrajectoryField position_fields[2][3] = {
        {BinaryTrajectoryField::position1_x,
         BinaryTrajectoryField::position1_y,
         BinaryTrajectoryField::position1_z},
        {BinaryTrajectoryField::position2_x,
         BinaryTrajectoryField::position2_y,
         BinaryTrajectoryField::position2_z}};
    const BinaryTrajectoryField velocity_fields[2][3] = {
        {BinaryTrajectoryField::velocity1_x,
         BinaryTrajectoryField::velocity1_y,
         BinaryTrajectoryField::velocity1_z},
        {BinaryTrajectoryField::velocity2_x,
         BinaryTrajectoryField::velocity2_y,
         BinaryTrajectoryField::velocity2_z}};
    Real a[3], b[3], c[3];
    for (int axis = 0; axis < 3; ++axis) {
        const auto pf = position_fields[hole][axis];
        const auto vf = velocity_fields[hole][axis];
        const Real y0 = values(table, pf)[lower];
        const Real y1 = values(table, pf)[upper];
        const Real d0 = values(table, vf)[lower];
        const Real d1 = values(table, vf)[upper];
        a[axis] = Real(6) * (y0 - y1) / dt +
                  Real(3) * (d0 + d1);
        b[axis] = Real(6) * (y1 - y0) / dt -
                  Real(4) * d0 - Real(2) * d1;
        c[axis] = d0;
    }
    Real p3 = Real(0), p2 = Real(0), p1 = Real(0), p0 = Real(0);
    for (int axis = 0; axis < 3; ++axis) {
        p3 += Real(2) * a[axis] * a[axis];
        p2 += Real(3) * a[axis] * b[axis];
        p1 += b[axis] * b[axis] + Real(2) * a[axis] * c[axis];
        p0 += b[axis] * c[axis];
    }
    std::vector<Real> partition = {Real(0), Real(1)};
    const Real qa = Real(3) * p3;
    const Real qb = Real(2) * p2;
    const Real qc = p1;
    const Real coefficient_scale = std::max<Real>(
        Real(1), std::max({std::abs(qa), std::abs(qb), std::abs(qc)}));
    const Real zero = Real(64) * std::numeric_limits<Real>::epsilon() *
                      coefficient_scale;
    if (std::abs(qa) > zero) {
        const Real discriminant = qb * qb - Real(4) * qa * qc;
        if (discriminant >= Real(0)) {
            const Real root = std::sqrt(discriminant);
            const Real u1 = (-qb - root) / (Real(2) * qa);
            const Real u2 = (-qb + root) / (Real(2) * qa);
            if (u1 > Real(0) && u1 < Real(1)) partition.push_back(u1);
            if (u2 > Real(0) && u2 < Real(1)) partition.push_back(u2);
        }
    } else if (std::abs(qb) > zero) {
        const Real u = -qc / qb;
        if (u > Real(0) && u < Real(1)) partition.push_back(u);
    }
    std::sort(partition.begin(), partition.end());
    std::vector<Real> candidates = partition;
    for (size_t i = 0; i + 1 < partition.size(); ++i) {
        Real left = partition[i];
        Real right = partition[i + 1];
        Real fl = cubic_value(p3, p2, p1, p0, left);
        Real fr = cubic_value(p3, p2, p1, p0, right);
        if (std::abs(fl) <= zero) candidates.push_back(left);
        if (std::abs(fr) <= zero) candidates.push_back(right);
        if ((fl < Real(0) && fr > Real(0)) ||
            (fl > Real(0) && fr < Real(0))) {
            for (int iteration = 0; iteration < 80; ++iteration) {
                const Real middle = Real(0.5) * (left + right);
                const Real fm = cubic_value(p3, p2, p1, p0, middle);
                if ((fl <= Real(0) && fm >= Real(0)) ||
                    (fl >= Real(0) && fm <= Real(0))) {
                    right = middle;
                    fr = fm;
                } else {
                    left = middle;
                    fl = fm;
                }
            }
            candidates.push_back(Real(0.5) * (left + right));
        }
    }
    Real maximum_speed2 = Real(0);
    for (Real u : candidates) {
        Real speed2 = Real(0);
        for (int axis = 0; axis < 3; ++axis) {
            const Real velocity = (a[axis] * u + b[axis]) * u + c[axis];
            speed2 += velocity * velocity;
        }
        maximum_speed2 = std::max(maximum_speed2, speed2);
    }
    if (!(maximum_speed2 < maximum_valid_boost_speed2())) {
        throw std::runtime_error(
            "cubic-Hermite trajectory contains a between-sample boost too close to or above light speed for the selected precision");
    }
}

size_t validate_and_canonicalize(HostTrajectoryTable& table,
                                 Real tolerance,
                                 BinaryTrajectoryMetadata& metadata) {
    const size_t count = table.time.size();
    if (count < 2) {
        throw std::runtime_error(
            "binary trajectory requires at least two time samples");
    }
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "binary trajectory sample count exceeds device index range");
    }
    for (const std::vector<Real>& field : table.field) {
        if (field.size() != count) {
            throw std::runtime_error(
                "binary trajectory host fields have inconsistent lengths");
        }
    }

    for (size_t i = 0; i < count; ++i) {
        require_finite(table.time[i], "time");
        if (i > 0 && !(table.time[i] > table.time[i - 1])) {
            throw std::runtime_error(
                "binary trajectory times must be finite and strictly increasing");
        }
        for (int field = 0; field < binary_trajectory_field_count; ++field) {
            require_finite(table.field[static_cast<size_t>(field)][i],
                           "field " + std::to_string(field));
        }
    }

    std::vector<Real>& weight =
        values(table, BinaryTrajectoryField::merger_weight);
    Real previous_weight = Real(0);
    for (size_t i = 0; i < count; ++i) {
        Real& w = weight[i];
        if (w < Real(0) || w > Real(1)) {
            throw std::runtime_error(
                "binary trajectory merger_weight must lie exactly in [0,1]");
        }
        // Endpoints are an exact schema contract.  A smooth window may
        // legitimately contain values one ULP from 0 or 1; fuzzy promotion
        // would truncate the transition and misidentify the remnant tail.
        if (i > 0 && w < previous_weight) {
            if (close_scalar(w, previous_weight, tolerance)) {
                w = previous_weight;
            } else {
                throw std::runtime_error(
                    "binary trajectory merger_weight must be nondecreasing");
            }
        }
        previous_weight = w;
    }

    for (size_t i = 0; i < count; ++i) {
        const Real m1 = values(table, BinaryTrajectoryField::mass1)[i];
        const Real m2 = values(table, BinaryTrajectoryField::mass2)[i];
        if (m1 < Real(0) || m2 < Real(0) || !(m1 + m2 > Real(0))) {
            throw std::runtime_error(
                "binary trajectory masses must be nonnegative with positive total mass");
        }
        for (int hole = 0; hole < 2; ++hole) {
            const Vec3<Real> velocity = velocity_at(table, hole, i);
            if (!(dot(velocity, velocity) < maximum_valid_boost_speed2())) {
                throw std::runtime_error(
                    "binary trajectory contains a hole velocity too close to or above light speed for the selected precision");
            }
            // A Kerr bound applies to each actual inspiral hole.  It does not
            // apply to an individual half-mass SKS term during/remnant collapse.
            if (weight[i] == Real(0)) {
                const Real mass = hole == 0 ? m1 : m2;
                const Real bound_slack = tolerance *
                    std::max<Real>(Real(1), mass);
                if (norm(kerr_a_at(table, hole, i)) > mass + bound_slack) {
                    throw std::runtime_error(
                        "binary trajectory violates |a|<=m before merger");
                }
            }
        }
    }
    if (metadata.interpolation == "cubic_hermite_position_velocity") {
        for (size_t i = 0; i + 1 < count; ++i) {
            validate_hermite_segment_speed(table, i, 0);
            validate_hermite_segment_speed(table, i, 1);
        }
    }

    size_t first_postmerger = count;
    for (size_t i = 0; i < count; ++i) {
        if (weight[i] == Real(1)) {
            first_postmerger = i;
            break;
        }
    }
    metadata.has_merger_transition =
        std::any_of(weight.begin(), weight.end(),
                    [](Real w) { return w > Real(0); });
    if (metadata.has_declared_transition_endpoints &&
        !metadata.has_merger_transition) {
        throw std::runtime_error(
            "declared binary trajectory transition endpoints require a nonzero merger_weight transition");
    }
    if (metadata.has_merger_transition) {
        size_t transition_start = 0;
        while (transition_start < count && weight[transition_start] == Real(0)) {
            ++transition_start;
        }
        size_t transition_end = transition_start;
        while (transition_end + 1 < count &&
               weight[transition_end] < Real(1)) {
            ++transition_end;
        }
        if (metadata.has_declared_transition_endpoints) {
            const auto matched_time_index = [&](Real declared,
                                                const char* label) {
                const auto iterator = std::lower_bound(
                    table.time.begin(), table.time.end(), declared);
                size_t candidate = static_cast<size_t>(
                    std::distance(table.time.begin(), iterator));
                if (candidate == count ||
                    table.time[candidate] != declared) {
                    if (candidate == 0 ||
                        table.time[candidate - 1] != declared) {
                        throw std::runtime_error(
                            std::string("declared binary trajectory ") +
                            label + " is not an explicit table sample");
                    }
                    --candidate;
                }
                return candidate;
            };
            if (!std::isfinite(static_cast<double>(
                    metadata.transition_start_time)) ||
                !std::isfinite(static_cast<double>(
                    metadata.transition_end_time)) ||
                !(metadata.transition_start_time <
                  metadata.transition_end_time) ||
                metadata.transition_start_time < table.time.front() ||
                metadata.transition_end_time > table.time.back()) {
                throw std::runtime_error(
                    "declared binary trajectory transition endpoints are invalid");
            }
            const size_t declared_start = matched_time_index(
                metadata.transition_start_time, "transition_start");
            const size_t declared_end = matched_time_index(
                metadata.transition_end_time, "transition_end");
            if (weight[declared_start] != Real(0) ||
                weight[declared_end] != Real(1) ||
                declared_start + 1 >= count ||
                !(weight[declared_start + 1] > Real(0)) ||
                declared_end == 0 ||
                !(weight[declared_end - 1] < Real(1)) ||
                declared_end != first_postmerger) {
                throw std::runtime_error(
                    "declared binary trajectory transition endpoints disagree with merger_weight");
            }
        } else {
            // Older/generic native and legacy tables do not carry exact
            // mathematical endpoints.  Retain a clearly approximate fallback
            // based on the first positive and first exact-unity samples.
            metadata.transition_start_time = table.time[transition_start];
            metadata.transition_end_time = table.time[transition_end];
        }
    }

    if (first_postmerger == count) {
        metadata.has_exact_postmerger_tail = false;
        return first_postmerger;
    }

    metadata.postmerger_start_time = table.time[first_postmerger];
    const Real comparison_tolerance = tolerance * Real(10);
    const Real reference_total_mass =
        values(table, BinaryTrajectoryField::mass1)[first_postmerger] +
        values(table, BinaryTrajectoryField::mass2)[first_postmerger];
    const Vec3<Real> reference_position = average_vector(
        position_at(table, 0, first_postmerger),
        position_at(table, 1, first_postmerger));
    const Vec3<Real> reference_velocity = average_vector(
        velocity_at(table, 0, first_postmerger),
        velocity_at(table, 1, first_postmerger));
    const Vec3<Real> reference_kerr_a = average_vector(
        kerr_a_at(table, 0, first_postmerger),
        kerr_a_at(table, 1, first_postmerger));
    const Real remnant_bound_slack = comparison_tolerance *
        std::max<Real>(Real(1), reference_total_mass);
    if (norm(reference_kerr_a) >
        reference_total_mass + remnant_bound_slack) {
        throw std::runtime_error(
            "postmerger trajectory violates the remnant Kerr bound |a_f|<=M_f");
    }

    for (size_t i = first_postmerger; i < count; ++i) {
        if (weight[i] != Real(1)) {
            throw std::runtime_error(
                "binary trajectory leaves W=1 after entering postmerger");
        }
        const Real m1 = values(table, BinaryTrajectoryField::mass1)[i];
        const Real m2 = values(table, BinaryTrajectoryField::mass2)[i];
        if (!close_scalar(m1, m2, comparison_tolerance) ||
            !close_scalar(m1 + m2, reference_total_mass,
                          comparison_tolerance)) {
            throw std::runtime_error(
                "postmerger SKS terms must have equal, constant half masses");
        }
        const Vec3<Real> p1 = position_at(table, 0, i);
        const Vec3<Real> p2 = position_at(table, 1, i);
        const Vec3<Real> v1 = velocity_at(table, 0, i);
        const Vec3<Real> v2 = velocity_at(table, 1, i);
        const Vec3<Real> a1 = kerr_a_at(table, 0, i);
        const Vec3<Real> a2 = kerr_a_at(table, 1, i);
        const Real dt = table.time[i] - table.time[first_postmerger];
        const Vec3<Real> expected_position =
            reference_position + reference_velocity * dt;
        if (!close_vector(p1, p2, comparison_tolerance) ||
            !close_vector(average_vector(p1, p2), expected_position,
                          comparison_tolerance) ||
            !close_vector(v1, v2, comparison_tolerance) ||
            !close_vector(average_vector(v1, v2), reference_velocity,
                          comparison_tolerance) ||
            !close_vector(a1, a2, comparison_tolerance) ||
            !close_vector(average_vector(a1, a2), reference_kerr_a,
                          comparison_tolerance)) {
            throw std::runtime_error(
                "postmerger terms must share one uniformly moving Kerr remnant");
        }
    }

    // Canonicalize accepted roundoff so adding the two rank-one SKS terms is
    // algebraically, not merely approximately, the single remnant Kerr term.
    for (size_t i = first_postmerger; i < count; ++i) {
        const Real dt = table.time[i] - table.time[first_postmerger];
        const Vec3<Real> position =
            reference_position + reference_velocity * dt;
        values(table, BinaryTrajectoryField::mass1)[i] =
            Real(0.5) * reference_total_mass;
        values(table, BinaryTrajectoryField::mass2)[i] =
            Real(0.5) * reference_total_mass;
        set_position_at(table, 0, i, position);
        set_position_at(table, 1, i, position);
        set_velocity_at(table, 0, i, reference_velocity);
        set_velocity_at(table, 1, i, reference_velocity);
        set_kerr_a_at(table, 0, i, reference_kerr_a);
        set_kerr_a_at(table, 1, i, reference_kerr_a);
        weight[i] = Real(1);
    }
    metadata.has_exact_postmerger_tail =
        first_postmerger + 1 < count;
    return first_postmerger;
}

HostTrajectoryTable load_native(H5::H5File& file,
                                const BinaryTrajectoryLoadOptions& options,
                                BinaryTrajectoryMetadata& metadata) {
    if (!link_exists(file, "/trajectory")) {
        throw std::runtime_error(
            "native binary trajectory is missing /trajectory group");
    }
    H5::Group group = file.openGroup("/trajectory");
    metadata.schema = read_string_attribute(group, "schema");
    if (metadata.schema != binary_trajectory_schema_name) {
        throw std::runtime_error(
            "unsupported native binary trajectory schema '" +
            metadata.schema + "'");
    }
    const int schema_version = read_integer_attribute(group, "schema_version");
    if (schema_version != binary_trajectory_schema_version) {
        throw std::runtime_error(
            "unsupported native binary trajectory schema_version " +
            std::to_string(schema_version));
    }
    metadata.units = read_string_attribute(group, "units");
    metadata.position_gauge =
        read_string_attribute(group, "position_gauge");
    metadata.spin_convention =
        read_string_attribute(group, "spin_convention");
    metadata.interpolation =
        read_string_attribute(group, "interpolation");
    metadata.interpolation = lowercase_copy(metadata.interpolation);
    metadata.generator = read_string_attribute(group, "generator");
    metadata.generator_version = read_optional_string_attribute(
        group, "generator_version");
    metadata.merger_reach_contract = read_optional_string_attribute(
        group, "merger_reach_contract");
    metadata.trajectory_model =
        read_string_attribute(group, "trajectory_model");
    metadata.pn_terms = read_string_attribute(group, "pn_terms");
    metadata.merger_separation_reached = read_optional_integer_attribute(
        group, "merger_separation_reached", -1);
    metadata.trajectory_status = read_optional_string_attribute(
        group, "trajectory_status");
    metadata.worldline_velocity_consistent =
        read_integer_attribute(group, "worldline_velocity_consistent");
    metadata.boost_velocity_model = lowercase_copy(
        read_string_attribute(group, "boost_velocity_model"));
    H5::Group root = file.openGroup("/");
    metadata.source_verified = read_optional_integer_attribute(
        root, "source_verified", -1);
    metadata.source_doi =
        read_optional_string_attribute(root, "source_doi");
    metadata.pn_4pn_scope =
        read_optional_string_attribute(root, "pn_4pn_scope");
    if (link_exists(file, "/provenance")) {
        H5::Group provenance = file.openGroup("/provenance");
        metadata.source_verification = read_optional_string_attribute(
            provenance, "source_verification");
        metadata.upstream_cbwaves_sha256 = read_optional_string_attribute(
            provenance, "upstream_cbwaves_sha256");
        metadata.patched_cbwaves_sha256 = read_optional_string_attribute(
            provenance, "patched_cbwaves_sha256");
    }
    if (metadata.generator.empty() || metadata.trajectory_model.empty() ||
        metadata.pn_terms.empty() || metadata.boost_velocity_model.empty()) {
        throw std::runtime_error(
            "native trajectory provenance and velocity-semantic attributes may not be empty");
    }
    if (metadata.source_verified < -1 || metadata.source_verified > 1) {
        throw std::runtime_error(
            "native source_verified must be scalar 0 or 1 when present");
    }
    if (metadata.merger_separation_reached < -1 ||
        metadata.merger_separation_reached > 1) {
        throw std::runtime_error(
            "native merger_separation_reached must be scalar 0 or 1 when present");
    }
    if (lowercase_copy(metadata.spin_convention) != "kerr_a") {
        throw std::runtime_error(
            "native spin_convention must be 'kerr_a' (S/M), not chi or S");
    }
    const std::string interpolation = metadata.interpolation;
    if (interpolation != "linear" &&
        interpolation != "cubic_hermite_position_velocity") {
        throw std::runtime_error(
            "native trajectory interpolation must be 'linear' or 'cubic_hermite_position_velocity'");
    }
    if (metadata.worldline_velocity_consistent < 0 ||
        metadata.worldline_velocity_consistent > 1) {
        throw std::runtime_error(
            "native worldline_velocity_consistent must be scalar 0 or 1");
    }
    if (interpolation == "cubic_hermite_position_velocity" &&
        (metadata.worldline_velocity_consistent != 1 ||
         metadata.boost_velocity_model != "derivative_of_position")) {
        throw std::runtime_error(
            "cubic Hermite trajectories require worldline_velocity_consistent=1 and boost_velocity_model='derivative_of_position'");
    }
    if (interpolation == "linear" &&
        metadata.worldline_velocity_consistent == 1) {
        throw std::runtime_error(
            "linear interpolation cannot claim worldline_velocity_consistent=1 because position and velocity are interpolated independently");
    }
    if (metadata.worldline_velocity_consistent == 0 &&
        metadata.boost_velocity_model == "derivative_of_position") {
        throw std::runtime_error(
            "native trajectory velocity semantic attributes are inconsistent");
    }
    if (metadata.worldline_velocity_consistent == 1 &&
        !metadata.boost_velocity_model.empty() &&
        metadata.boost_velocity_model != "derivative_of_position") {
        throw std::runtime_error(
            "native trajectory velocity semantic attributes are inconsistent");
    }
    if (metadata.units != "G=c=M_ref=1") {
        throw std::runtime_error(
            "native trajectory units must be exactly 'G=c=M_ref=1'");
    }
    if (lowercase_copy(metadata.position_gauge) != "harmonic") {
        throw std::runtime_error(
            "native paper-CBwaves trajectories require position_gauge='harmonic'");
    }

    const NumericDataset time_dataset =
        read_numeric_dataset(file, "/trajectory/t");
    if (time_dataset.dimensions.size() != 1 ||
        time_dataset.dimensions[0] < 2) {
        throw std::runtime_error(
            "/trajectory/t must be a one-dimensional array with N>=2");
    }
    const size_t count = static_cast<size_t>(time_dataset.dimensions[0]);
    const NumericDataset mass =
        read_numeric_dataset(file, "/trajectory/mass");
    const NumericDataset position =
        read_numeric_dataset(file, "/trajectory/position");
    const NumericDataset velocity =
        read_numeric_dataset(file, "/trajectory/velocity");
    const NumericDataset kerr_a =
        read_numeric_dataset(file, "/trajectory/kerr_a");
    const NumericDataset merger_weight =
        read_numeric_dataset(file, "/trajectory/merger_weight");
    const bool has_spin_chi = link_exists(file, "/trajectory/spin_chi");
    NumericDataset spin_chi;
    if (has_spin_chi) {
        spin_chi = read_numeric_dataset(file, "/trajectory/spin_chi");
    }
    require_dimensions(mass, {static_cast<hsize_t>(count), 2},
                       "/trajectory/mass");
    require_dimensions(position, {static_cast<hsize_t>(count), 2, 3},
                       "/trajectory/position");
    require_dimensions(velocity, {static_cast<hsize_t>(count), 2, 3},
                       "/trajectory/velocity");
    require_dimensions(kerr_a, {static_cast<hsize_t>(count), 2, 3},
                       "/trajectory/kerr_a");
    require_dimensions(merger_weight, {static_cast<hsize_t>(count)},
                       "/trajectory/merger_weight");
    if (has_spin_chi) {
        require_dimensions(spin_chi,
                           {static_cast<hsize_t>(count), 2, 3},
                           "/trajectory/spin_chi");
        for (size_t i = 0; i < count; ++i) {
            for (size_t hole = 0; hole < 2; ++hole) {
                const double m = mass.values[2 * i + hole];
                for (size_t axis = 0; axis < 3; ++axis) {
                    const size_t component = i * 6 + hole * 3 + axis;
                    const double chi = spin_chi.values[component];
                    const double a = kerr_a.values[component];
                    const double expected = m * chi;
                    const double scale = std::max(
                        1.0, std::max(std::abs(a), std::abs(expected)));
                    if (!std::isfinite(chi) || !std::isfinite(a) ||
                        std::abs(a - expected) >
                            static_cast<double>(options.relative_tolerance) *
                                scale) {
                        throw std::runtime_error(
                            "/trajectory/spin_chi is inconsistent with kerr_a=mass*spin_chi");
                    }
                }
            }
        }
    }

    HostTrajectoryTable table;
    table.time.resize(count);
    allocate_fields(table, count);
    const Real geometric_scale = options.geometric_unit_scale;
    const bool has_transition_start =
        link_exists(file, "/remnant/transition_start");
    const bool has_transition_end =
        link_exists(file, "/remnant/transition_end");
    if (has_transition_start != has_transition_end) {
        throw std::runtime_error(
            "native trajectory must provide both /remnant/transition_start and /remnant/transition_end or neither");
    }
    if (has_transition_start) {
        metadata.transition_start_time = read_scalar(
            file, "/remnant/transition_start", geometric_scale);
        metadata.transition_end_time = read_scalar(
            file, "/remnant/transition_end", geometric_scale);
        metadata.has_declared_transition_endpoints = true;
    } else if (!metadata.merger_reach_contract.empty()) {
        throw std::runtime_error(
            "contracted paper trajectory lacks exact /remnant transition endpoints");
    }
    for (size_t i = 0; i < count; ++i) {
        table.time[i] =
            static_cast<Real>(time_dataset.values[i]) * geometric_scale;
        values(table, BinaryTrajectoryField::mass1)[i] =
            static_cast<Real>(mass.values[2 * i]) * geometric_scale;
        values(table, BinaryTrajectoryField::mass2)[i] =
            static_cast<Real>(mass.values[2 * i + 1]) * geometric_scale;
        for (int hole = 0; hole < 2; ++hole) {
            const size_t base = i * 6 + static_cast<size_t>(hole) * 3;
            const Vec3<Real> p(
                static_cast<Real>(position.values[base]) * geometric_scale,
                static_cast<Real>(position.values[base + 1]) * geometric_scale,
                static_cast<Real>(position.values[base + 2]) * geometric_scale);
            const Vec3<Real> v(
                static_cast<Real>(velocity.values[base]),
                static_cast<Real>(velocity.values[base + 1]),
                static_cast<Real>(velocity.values[base + 2]));
            const Vec3<Real> a(
                static_cast<Real>(kerr_a.values[base]) * geometric_scale,
                static_cast<Real>(kerr_a.values[base + 1]) * geometric_scale,
                static_cast<Real>(kerr_a.values[base + 2]) * geometric_scale);
            set_position_at(table, hole, i, p);
            set_velocity_at(table, hole, i, v);
            set_kerr_a_at(table, hole, i, a);
        }
        values(table, BinaryTrajectoryField::merger_weight)[i] =
            static_cast<Real>(merger_weight.values[i]);
    }
    return table;
}

int count_existing(H5::H5File& file,
                   const std::vector<std::string>& paths) {
    int count = 0;
    for (const std::string& path : paths) {
        if (link_exists(file, path)) ++count;
    }
    return count;
}

void require_all_or_none(H5::H5File& file,
                         const std::vector<std::string>& paths,
                         const std::string& description) {
    const int present = count_existing(file, paths);
    if (present != 0 && present != static_cast<int>(paths.size())) {
        throw std::runtime_error("incomplete " + description +
                                 " in legacy Combi--Ressler trajectory");
    }
}

void load_legacy_vector_components(
    H5::H5File& file, HostTrajectoryTable& table,
    const std::array<const char*, 6>& paths,
    const std::array<BinaryTrajectoryField, 6>& fields,
    Real scale) {
    const size_t count = table.time.size();
    for (size_t component = 0; component < paths.size(); ++component) {
        values(table, fields[component]) =
            read_vector(file, paths[component], count, scale);
    }
}

void infer_legacy_merger_weight(
    H5::H5File& file, HostTrajectoryTable& table,
    bool has_dynamic_masses, bool has_initial_scalars,
    Real initial_mass1, Real initial_mass2,
    Real geometric_scale, Real tolerance,
    BinaryTrajectoryMetadata& metadata) {
    const size_t count = table.time.size();
    std::vector<Real>& weight =
        values(table, BinaryTrajectoryField::merger_weight);
    const bool has_long_name = link_exists(file, "merger_weight");
    const bool has_short_name = link_exists(file, "W");
    if (has_long_name && has_short_name) {
        throw std::runtime_error(
            "legacy trajectory contains both merger_weight and W datasets");
    }
    if (has_long_name || has_short_name) {
        weight = read_vector(file, has_long_name ? "merger_weight" : "W",
                             count);
        return;
    }

    bool inferred_from_mass = false;
    if (has_dynamic_masses) {
        const Real initial_total = has_initial_scalars ?
            initial_mass1 + initial_mass2 :
            values(table, BinaryTrajectoryField::mass1).front() +
                values(table, BinaryTrajectoryField::mass2).front();
        const Real final_total =
            values(table, BinaryTrajectoryField::mass1).back() +
            values(table, BinaryTrajectoryField::mass2).back();
        const Real denominator = initial_total - final_total;
        if (std::abs(denominator) >
            tolerance * std::max<Real>(Real(1), std::abs(initial_total))) {
            for (size_t i = 0; i < count; ++i) {
                const Real total =
                    values(table, BinaryTrajectoryField::mass1)[i] +
                    values(table, BinaryTrajectoryField::mass2)[i];
                weight[i] = (initial_total - total) / denominator;
            }
            inferred_from_mass = true;
            metadata.warnings.push_back(
                "legacy merger_weight inferred from total-mass transition");
        }
    }

    if (!inferred_from_mass && link_exists(file, "t_postmerger")) {
        const Real postmerger_time =
            read_scalar(file, "t_postmerger", geometric_scale);
        for (size_t i = 0; i < count; ++i) {
            weight[i] = table.time[i] >= postmerger_time ? Real(1) : Real(0);
        }
        metadata.warnings.push_back(
            "legacy file lacks reconstructible smooth W; t_postmerger gives a step marker");
    }
}

bool read_optional_remnant_spin(H5::H5File& file, Real geometric_scale,
                                Vec3<Real>& spin) {
    const std::vector<std::string> paths = {
        "a_x_remnant", "a_y_remnant", "a_z_remnant"};
    require_all_or_none(file, paths, "remnant-spin datasets");
    if (count_existing(file, paths) == 0) return false;
    spin = Vec3<Real>(read_scalar(file, paths[0], geometric_scale),
                      read_scalar(file, paths[1], geometric_scale),
                      read_scalar(file, paths[2], geometric_scale));
    return true;
}

void repair_legacy_merger(
    H5::H5File& file, HostTrajectoryTable& table,
    bool has_initial_scalars, Real initial_mass1, Real initial_mass2,
    Real geometric_scale, Real tolerance,
    BinaryTrajectoryMetadata& metadata) {
    const size_t count = table.time.size();
    std::vector<Real>& weight =
        values(table, BinaryTrajectoryField::merger_weight);
    size_t first_postmerger = count;
    for (size_t i = 0; i < count; ++i) {
        if (weight[i] == Real(1)) {
            first_postmerger = i;
            break;
        }
    }
    if (first_postmerger == count) return;

    const size_t last = count - 1;
    const Real comparison_tolerance = tolerance * Real(20);
    if (!close_vector(position_at(table, 0, last),
                      position_at(table, 1, last), comparison_tolerance) ||
        !close_vector(velocity_at(table, 0, last),
                      velocity_at(table, 1, last), comparison_tolerance) ||
        !close_vector(kerr_a_at(table, 0, last),
                      kerr_a_at(table, 1, last), comparison_tolerance)) {
        throw std::runtime_error(
            "legacy trajectory claims postmerger but its final SKS terms do not coincide");
    }

    const Real final_mass =
        values(table, BinaryTrajectoryField::mass1)[last] +
        values(table, BinaryTrajectoryField::mass2)[last];
    if (!(final_mass > Real(0))) {
        throw std::runtime_error(
            "legacy trajectory has nonpositive remnant mass");
    }
    const Real start_mass1 = has_initial_scalars ? initial_mass1 :
        values(table, BinaryTrajectoryField::mass1).front();
    const Real start_mass2 = has_initial_scalars ? initial_mass2 :
        values(table, BinaryTrajectoryField::mass2).front();
    const Vec3<Real> final_position = average_vector(
        position_at(table, 0, first_postmerger),
        position_at(table, 1, first_postmerger));
    const Vec3<Real> final_velocity = average_vector(
        velocity_at(table, 0, last), velocity_at(table, 1, last));
    Vec3<Real> final_kerr_a = average_vector(
        kerr_a_at(table, 0, last), kerr_a_at(table, 1, last));
    Vec3<Real> metadata_kerr_a;
    if (read_optional_remnant_spin(file, geometric_scale,
                                   metadata_kerr_a)) {
        if (!close_vector(final_kerr_a, metadata_kerr_a,
                          comparison_tolerance)) {
            throw std::runtime_error(
                "legacy remnant-spin metadata disagrees with trajectory arrays");
        }
        final_kerr_a = metadata_kerr_a;
    }

    // Correct the published unequal-mass branch throughout the smooth
    // transition.  Both terms approach Mf/2, while each carries the full Kerr
    // a-vector; only then is their sum exactly one remnant Kerr-Schild term.
    for (size_t i = 0; i < count; ++i) {
        Real w = weight[i];
        weight[i] = w;
        values(table, BinaryTrajectoryField::mass1)[i] =
            start_mass1 * (Real(1) - w) + Real(0.5) * final_mass * w;
        values(table, BinaryTrajectoryField::mass2)[i] =
            start_mass2 * (Real(1) - w) + Real(0.5) * final_mass * w;
    }
    for (size_t i = first_postmerger; i < count; ++i) {
        const Real dt = table.time[i] - table.time[first_postmerger];
        const Vec3<Real> position = final_position + final_velocity * dt;
        values(table, BinaryTrajectoryField::mass1)[i] =
            Real(0.5) * final_mass;
        values(table, BinaryTrajectoryField::mass2)[i] =
            Real(0.5) * final_mass;
        set_position_at(table, 0, i, position);
        set_position_at(table, 1, i, position);
        set_velocity_at(table, 0, i, final_velocity);
        set_velocity_at(table, 1, i, final_velocity);
        set_kerr_a_at(table, 0, i, final_kerr_a);
        set_kerr_a_at(table, 1, i, final_kerr_a);
        weight[i] = Real(1);
    }
    metadata.repaired_legacy_merger = true;
    metadata.warnings.push_back(
        "legacy merger canonicalized to two coincident Mf/2 terms and a uniform remnant worldline");
}

HostTrajectoryTable load_combi_ressler(
    H5::H5File& file, const BinaryTrajectoryLoadOptions& options,
    BinaryTrajectoryMetadata& metadata) {
    const Real geometric_scale = options.geometric_unit_scale;
    const NumericDataset time_dataset = read_numeric_dataset(file, "t");
    if (time_dataset.dimensions.size() != 1 ||
        time_dataset.dimensions[0] < 2) {
        throw std::runtime_error(
            "legacy t must be a one-dimensional array with N>=2");
    }
    const size_t count = static_cast<size_t>(time_dataset.dimensions[0]);
    HostTrajectoryTable table;
    table.time.resize(count);
    allocate_fields(table, count);
    for (size_t i = 0; i < count; ++i) {
        table.time[i] =
            static_cast<Real>(time_dataset.values[i]) * geometric_scale;
    }

    const std::array<const char*, 6> position_paths = {
        "x1", "y1", "z1", "x2", "y2", "z2"};
    const std::array<BinaryTrajectoryField, 6> position_fields = {
        BinaryTrajectoryField::position1_x,
        BinaryTrajectoryField::position1_y,
        BinaryTrajectoryField::position1_z,
        BinaryTrajectoryField::position2_x,
        BinaryTrajectoryField::position2_y,
        BinaryTrajectoryField::position2_z};
    const std::array<const char*, 6> velocity_paths = {
        "vx1", "vy1", "vz1", "vx2", "vy2", "vz2"};
    const std::array<BinaryTrajectoryField, 6> velocity_fields = {
        BinaryTrajectoryField::velocity1_x,
        BinaryTrajectoryField::velocity1_y,
        BinaryTrajectoryField::velocity1_z,
        BinaryTrajectoryField::velocity2_x,
        BinaryTrajectoryField::velocity2_y,
        BinaryTrajectoryField::velocity2_z};
    load_legacy_vector_components(file, table, position_paths,
                                  position_fields, geometric_scale);
    load_legacy_vector_components(file, table, velocity_paths,
                                  velocity_fields, Real(1));

    const std::vector<std::string> dynamic_mass_paths = {
        "m1_full", "m2_full"};
    require_all_or_none(file, dynamic_mass_paths,
                        "m1_full/m2_full datasets");
    const bool has_dynamic_masses =
        count_existing(file, dynamic_mass_paths) == 2;
    const bool has_m1 = link_exists(file, "m1");
    const bool has_m2 = link_exists(file, "m2");
    if (has_m1 != has_m2) {
        throw std::runtime_error(
            "legacy trajectory must provide both scalar m1 and m2");
    }
    const bool has_initial_scalars = has_m1 && has_m2;
    Real initial_mass1 = Real(0);
    Real initial_mass2 = Real(0);
    if (has_initial_scalars) {
        initial_mass1 = read_scalar(file, "m1", geometric_scale);
        initial_mass2 = read_scalar(file, "m2", geometric_scale);
    }
    if (has_dynamic_masses) {
        values(table, BinaryTrajectoryField::mass1) =
            read_vector(file, "m1_full", count, geometric_scale);
        values(table, BinaryTrajectoryField::mass2) =
            read_vector(file, "m2_full", count, geometric_scale);
    } else {
        if (!has_initial_scalars) {
            throw std::runtime_error(
                "legacy trajectory needs m1/m2 or m1_full/m2_full");
        }
        std::fill(values(table, BinaryTrajectoryField::mass1).begin(),
                  values(table, BinaryTrajectoryField::mass1).end(),
                  initial_mass1);
        std::fill(values(table, BinaryTrajectoryField::mass2).begin(),
                  values(table, BinaryTrajectoryField::mass2).end(),
                  initial_mass2);
    }

    const std::vector<std::string> kerr_a_paths = {
        "a1x", "a1y", "a1z", "a2x", "a2y", "a2z"};
    const std::vector<std::string> chi_paths = {
        "s1x", "s1y", "s1z", "s2x", "s2y", "s2z"};
    require_all_or_none(file, kerr_a_paths, "a-vector datasets");
    require_all_or_none(file, chi_paths, "dimensionless-spin datasets");
    const bool has_kerr_a = count_existing(file, kerr_a_paths) == 6;
    const bool has_chi = count_existing(file, chi_paths) == 6;
    if (has_kerr_a == has_chi) {
        throw std::runtime_error(
            "legacy trajectory must provide exactly one complete spin convention: a1x...a2z or s1x...s2z");
    }
    const std::array<BinaryTrajectoryField, 6> spin_fields = {
        BinaryTrajectoryField::kerr_a1_x,
        BinaryTrajectoryField::kerr_a1_y,
        BinaryTrajectoryField::kerr_a1_z,
        BinaryTrajectoryField::kerr_a2_x,
        BinaryTrajectoryField::kerr_a2_y,
        BinaryTrajectoryField::kerr_a2_z};
    for (size_t component = 0; component < spin_fields.size(); ++component) {
        std::vector<Real> source = read_vector(
            file, has_kerr_a ? kerr_a_paths[component] : chi_paths[component],
            count, has_kerr_a ? geometric_scale : Real(1));
        if (!has_kerr_a) {
            const bool first_hole = component < 3;
            const std::vector<Real>& mass_values = values(
                table, first_hole ? BinaryTrajectoryField::mass1 :
                                    BinaryTrajectoryField::mass2);
            for (size_t i = 0; i < count; ++i) {
                source[i] *= mass_values[i];
            }
        }
        values(table, spin_fields[component]) = std::move(source);
    }

    infer_legacy_merger_weight(
        file, table, has_dynamic_masses, has_initial_scalars,
        initial_mass1, initial_mass2, geometric_scale,
        options.relative_tolerance, metadata);
    if (options.repair_combi_ressler_merger) {
        repair_legacy_merger(
            file, table, has_initial_scalars, initial_mass1, initial_mass2,
            geometric_scale, options.relative_tolerance, metadata);
    }

    metadata.schema = "combi_ressler_flat_hdf5";
    metadata.units = "legacy_geometric_units_scaled_by_loader";
    metadata.position_gauge = "legacy_file_unspecified";
    metadata.spin_convention = has_kerr_a ? "kerr_a" : "chi_converted_to_kerr_a";
    metadata.interpolation = "linear";
    metadata.generator = "Combi--Ressler trajectory_tools.py";
    metadata.worldline_velocity_consistent = 0;
    metadata.boost_velocity_model = "legacy_independent_velocity_columns";
    return table;
}

LoadedBinaryTrajectory materialize(
    HostTrajectoryTable table,
    const BinaryTrajectoryLoadOptions& options,
    BinaryTrajectoryMetadata metadata) {
    validate_and_canonicalize(table, options.relative_tolerance, metadata);
    const size_t count = table.time.size();
    using Provider = BinaryTrajectoryProvider<Real>;
    using MemorySpace = typename Provider::DeviceMemorySpace;
    Kokkos::View<Real*, MemorySpace> device_time(
        "binary_trajectory_time", count);
    Kokkos::View<Real*, MemorySpace> device_values(
        "binary_trajectory_fields",
        static_cast<size_t>(binary_trajectory_field_count) * count);
    auto host_time = Kokkos::create_mirror_view(device_time);
    auto host_values = Kokkos::create_mirror_view(device_values);
    for (size_t i = 0; i < count; ++i) {
        host_time(i) = table.time[i];
    }
    for (int field = 0; field < binary_trajectory_field_count; ++field) {
        for (size_t i = 0; i < count; ++i) {
            host_values(static_cast<size_t>(field) * count + i) =
                table.field[static_cast<size_t>(field)][i];
        }
    }
    Kokkos::deep_copy(device_time, host_time);
    Kokkos::deep_copy(device_values, host_values);

    LoadedBinaryTrajectory output;
    output.metadata = std::move(metadata);
    output.metadata.sample_count = count;
    output.metadata.time_min = table.time.front();
    output.metadata.time_max = table.time.back();
    const bool allow_future_extension =
        output.metadata.format == BinaryTrajectoryHdf5Format::native &&
        output.metadata.has_exact_postmerger_tail &&
        options.enable_future_postmerger_extension;
    output.metadata.future_postmerger_extension_enabled =
        allow_future_extension;
    if (output.metadata.format ==
            BinaryTrajectoryHdf5Format::combi_ressler &&
        output.metadata.has_exact_postmerger_tail &&
        options.enable_future_postmerger_extension) {
        output.metadata.warnings.push_back(
            "future remnant extrapolation is disabled for legacy Combi--Ressler files because algebraic repair cannot verify a shared worldline or kick; convert a verified trajectory to the native schema to enable it");
    }

    Provider& provider = output.provider;
    provider.mode = static_cast<int>(BinaryTrajectoryMode::tabulated);
    provider.interpolation = output.metadata.interpolation ==
            "cubic_hermite_position_velocity" ?
        static_cast<int>(BinaryTrajectoryInterpolation::
            cubic_hermite_position_velocity) :
        static_cast<int>(BinaryTrajectoryInterpolation::linear);
    provider.table_times = device_time;
    provider.table_values = device_values;
    provider.sample_count = static_cast<int>(count);
    provider.table_time_min = table.time.front();
    provider.table_time_max = table.time.back();
    provider.has_exact_postmerger_tail =
        output.metadata.has_exact_postmerger_tail ? 1 : 0;
    provider.enable_future_postmerger_extension =
        allow_future_extension ? 1 : 0;

    // Populate the retained legacy fields with the first tabulated state.  The
    // table mode does not read them, but diagnostics and old configuration
    // plumbing can continue to report meaningful initial values.
    provider.mass1 = values(table, BinaryTrajectoryField::mass1).front();
    provider.mass2 = values(table, BinaryTrajectoryField::mass2).front();
    provider.total_mass = provider.mass1 + provider.mass2;
    provider.reference_time = table.time.front();
    provider.kerr_a1 = kerr_a_at(table, 0, 0);
    provider.kerr_a2 = kerr_a_at(table, 1, 0);
    provider.reference_separation =
        norm(position_at(table, 0, 0) - position_at(table, 1, 0));
    const Vec3<Real> relative_position =
        position_at(table, 0, 0) - position_at(table, 1, 0);
    provider.reference_phase =
        std::atan2(relative_position.y, relative_position.x);
    provider.minimum_separation = Real(0);
    provider.inspiral_enabled = 0;
    provider.orbit_enabled = 0;
    return output;
}

} // namespace

LoadedBinaryTrajectory load_binary_trajectory_hdf5(
    const std::string& path, const BinaryTrajectoryLoadOptions& options) {
    if (path.empty()) {
        throw std::runtime_error("binary trajectory HDF5 path is empty");
    }
    if (!std::isfinite(static_cast<double>(options.geometric_unit_scale)) ||
        !(options.geometric_unit_scale > Real(0))) {
        throw std::runtime_error(
            "binary trajectory geometric_unit_scale must be finite and positive");
    }
    if (!std::isfinite(static_cast<double>(options.relative_tolerance)) ||
        !(options.relative_tolerance > Real(0)) ||
        !(options.relative_tolerance < Real(0.01))) {
        throw std::runtime_error(
            "binary trajectory relative_tolerance must lie in (0,0.01)");
    }

    H5::Exception::dontPrint();
    try {
        H5::H5File file(path, H5F_ACC_RDONLY);
        BinaryTrajectoryMetadata metadata;
        metadata.source_path = path;

        BinaryTrajectoryHdf5Format format = options.format;
        if (format == BinaryTrajectoryHdf5Format::automatic) {
            if (link_exists(file, "/trajectory") &&
                link_exists(file, "/trajectory/t")) {
                format = BinaryTrajectoryHdf5Format::native;
            } else if (link_exists(file, "t") &&
                       link_exists(file, "x1")) {
                format = BinaryTrajectoryHdf5Format::combi_ressler;
            } else {
                throw std::runtime_error(
                    "unable to identify binary trajectory HDF5 schema");
            }
        }
        metadata.format = format;
        HostTrajectoryTable table;
        if (format == BinaryTrajectoryHdf5Format::native) {
            table = load_native(file, options, metadata);
        } else if (format == BinaryTrajectoryHdf5Format::combi_ressler) {
            table = load_combi_ressler(file, options, metadata);
        } else {
            throw std::runtime_error(
                "invalid binary trajectory HDF5 format selection");
        }
        if (options.require_source_verified &&
            metadata.source_verified != 1) {
            throw std::runtime_error(
                "trajectory does not carry source_verified=1");
        }
        if (options.require_merger_separation_reached &&
            (metadata.merger_separation_reached != 1 ||
             metadata.trajectory_status !=
                 "merger_separation_reached")) {
            throw std::runtime_error(
                "paper trajectory is diagnostic/truncated and did not reach the requested merger separation");
        }
        const auto require_exact_metadata = [](
            const std::string& field, const std::string& actual,
            const std::string& required) {
            if (!required.empty() && actual != required) {
                throw std::runtime_error(
                    "trajectory " + field + " mismatch: expected '" +
                    required + "', got '" + actual + "'");
            }
        };
        require_exact_metadata("generator", metadata.generator,
                               options.required_generator);
        require_exact_metadata("generator_version",
                               metadata.generator_version,
                               options.required_generator_version);
        require_exact_metadata("merger_reach_contract",
                               metadata.merger_reach_contract,
                               options.required_merger_reach_contract);
        require_exact_metadata("trajectory_model", metadata.trajectory_model,
                               options.required_trajectory_model);
        require_exact_metadata("pn_terms", metadata.pn_terms,
                               options.required_pn_terms);
        require_exact_metadata("source_doi", metadata.source_doi,
                               options.required_source_doi);
        require_exact_metadata("source_verification",
                               metadata.source_verification,
                               options.required_source_verification);
        require_exact_metadata("upstream_cbwaves_sha256",
                               metadata.upstream_cbwaves_sha256,
                               options.required_upstream_cbwaves_sha256);
        require_exact_metadata("patched_cbwaves_sha256",
                               metadata.patched_cbwaves_sha256,
                               options.required_patched_cbwaves_sha256);
        return materialize(std::move(table), options, std::move(metadata));
    } catch (const H5::Exception& error) {
        throw std::runtime_error(
            "failed to load binary trajectory HDF5 '" + path + "': " +
            error.getDetailMsg());
    }
}

} // namespace kpolaris
