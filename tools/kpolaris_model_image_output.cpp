#include "kpolaris_model_image_output.hpp"

#include <cmath>
#include <stdexcept>

#include "common/types.hpp"

bool exceeds_positive_threshold(double value, kpolaris::DefaultReal threshold) {
    return threshold > kpolaris::DefaultReal(0) && value > static_cast<double>(threshold);
}

void update_max_abs(double value, int pixel, double& max_value, int& max_pixel) {
    if (!std::isfinite(value)) {
        return;
    }
    const double abs_value = std::abs(value);
    if (abs_value > max_value) {
        max_value = abs_value;
        max_pixel = pixel;
    }
}

DiagnosticSummary summarize_diagnostics(const Options& opt,
                                        const std::vector<double>& closure_x,
                                        const std::vector<double>& closure_k,
                                        const std::vector<double>& frame_error,
                                        const std::vector<double>& basis_identity_error,
                                        const std::vector<int>& reason) {
    DiagnosticSummary summary;
    for (int p = 0; p < opt.nx * opt.ny; ++p) {
        if (reason[p] != static_cast<int>(kpolaris::TerminationReason::reached_camera)) {
            continue;
        }
        update_max_abs(closure_x[p], p, summary.max_closure_x,
                       summary.max_closure_x_pixel);
        update_max_abs(closure_k[p], p, summary.max_closure_k,
                       summary.max_closure_k_pixel);
        update_max_abs(frame_error[p], p, summary.max_frame_error,
                       summary.max_frame_error_pixel);
        update_max_abs(basis_identity_error[p], p,
                       summary.max_basis_identity_error,
                       summary.max_basis_identity_error_pixel);

        const bool warn_closure_x =
            exceeds_positive_threshold(std::abs(closure_x[p]), opt.closure_x_warning);
        const bool warn_closure_k =
            exceeds_positive_threshold(std::abs(closure_k[p]), opt.closure_k_warning);
        const bool warn_frame =
            exceeds_positive_threshold(std::abs(frame_error[p]), opt.frame_error_warning);
        const bool warn_basis =
            exceeds_positive_threshold(std::abs(basis_identity_error[p]),
                                       opt.basis_identity_warning);
        summary.closure_x_warning_count += warn_closure_x ? 1 : 0;
        summary.closure_k_warning_count += warn_closure_k ? 1 : 0;
        summary.frame_error_warning_count += warn_frame ? 1 : 0;
        summary.basis_identity_warning_count += warn_basis ? 1 : 0;
        summary.any_warning_count +=
            (warn_closure_x || warn_closure_k || warn_frame || warn_basis) ? 1 : 0;
    }
    return summary;
}

int pixel_ix(int pixel, int nx) {
    return pixel >= 0 ? pixel % nx : -1;
}

int pixel_iy(int pixel, int nx) {
    return pixel >= 0 ? pixel / nx : -1;
}

void write_h5_scalar(H5::H5Object& obj, const std::string& name, double value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::NATIVE_DOUBLE, space);
    attr.write(H5::PredType::NATIVE_DOUBLE, &value);
}

void write_h5_int(H5::H5Object& obj, const std::string& name, int value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::NATIVE_INT, space);
    attr.write(H5::PredType::NATIVE_INT, &value);
}

void write_h5_string(H5::H5Object& obj, const std::string& name, const std::string& value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute attr = obj.createAttribute(name, type, space);
    const char* cstr = value.c_str();
    attr.write(type, &cstr);
}

void write_h5_dataset_2d(H5::Group& group, const std::string& name,
                         const std::vector<double>& values, int nx, int ny) {
    hsize_t dims[2] = {static_cast<hsize_t>(ny), static_cast<hsize_t>(nx)};
    H5::DataSpace space(2, dims);
    H5::DataSet ds = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(values.data(), H5::PredType::NATIVE_DOUBLE);
}

void write_h5_dataset_2d_int(H5::Group& group, const std::string& name,
                             const std::vector<int>& values, int nx, int ny) {
    hsize_t dims[2] = {static_cast<hsize_t>(ny), static_cast<hsize_t>(nx)};
    H5::DataSpace space(2, dims);
    H5::DataSet ds = group.createDataSet(name, H5::PredType::NATIVE_INT, space);
    ds.write(values.data(), H5::PredType::NATIVE_INT);
}

void write_h5_dataset_3d(H5::Group& group, const std::string& name,
                         const std::vector<double>& values,
                         int nplane, int nx, int ny) {
    const size_t expected = static_cast<size_t>(nplane) *
                            static_cast<size_t>(nx) * static_cast<size_t>(ny);
    if (values.size() != expected) {
        throw std::runtime_error("3D HDF5 dataset size does not match requested shape: " + name);
    }
    hsize_t dims[3] = {static_cast<hsize_t>(nplane),
                       static_cast<hsize_t>(ny), static_cast<hsize_t>(nx)};
    H5::DataSpace space(3, dims);
    H5::DataSet ds = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(values.data(), H5::PredType::NATIVE_DOUBLE);
}

void write_h5_dataset_1d(H5::Group& group, const std::string& name,
                         const std::vector<double>& values) {
    hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
    H5::DataSpace space(1, dims);
    H5::DataSet ds = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(values.data(), H5::PredType::NATIVE_DOUBLE);
}
