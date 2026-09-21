#pragma once

#include <H5Cpp.h>
#include <string>
#include "kpolaris_evpa.hpp"

// Shared metadata for the explicit host-side N/W output basis.
inline void write_kpolaris_camera_conventions(H5::H5Object& object,
                                             const std::string& evpa_0, bool trace = false) {
    auto text = [&](const char* name, const char* value) {
        H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
        H5::DataSpace scalar(H5S_SCALAR);
        auto attr = object.createAttribute(name, type, scalar);
        attr.write(type, std::string(value));
    };
    text("polarization_conventions_version", "2");
    text("metric_signature", "-+++");
    text("wavevector_direction", "future_directed; Pass A uses negative affine steps, Pass B positive");
    text("camera_azimuth_convention", "pinhole: native phi=0; parallel: center in Cartesian x-z plane; no camera roll");
    text("sky_orientation", "image north=up, east=left by convention; celestial position angle not assigned");
    text("screen_handedness", "right_handed (e1,e2,photon_direction) in the observer rest frame");
    text("stokes_definition", "I=<|E1|^2+|E2|^2>; Q=<|E1|^2-|E2|^2>; U=2Re<E1 E2*>; V=-2Im<E1 E2*>");
    text("electric_field_phase_convention", "Re[E exp(-i omega t)] in a local right-handed screen");
    text("stokes_basis_definition", evpa_0 == "N"
        ? "output basis (e2,-e1), a passive 90-degree rotation of the internal camera screen"
        : "output basis (e1,e2), the internal camera screen");
    text("evpa_definition", evpa_0 == "N"
        ? "0.5 atan2(U,Q), modulo pi; zero=image north, increasing toward image east in the distant-observer limit; undefined at Q=U=0"
        : "0.5 atan2(U,Q), modulo pi; zero=horizontal (W), increasing toward vertical (N) in the distant-observer limit; undefined at Q=U=0");
    text("image_pixel_order", "pixel=iy*nx+ix; increasing ix right, iy up");
    text("image_axis_geometry", "phi=0 distant-observer limit: +x=+e_phi, +y=-e_theta (projected coordinate +z)");
    text("pinhole_screen_geometry", "project tetrad e1,e2 perpendicular to each k, then Gram-Schmidt; gnomonic image chart");
    text("output_stokes_transform", evpa_0 == "N"
        ? "I,Q,U,V=(I_camera,-Q_camera,-U_camera,V_camera); passive 90-degree basis rotation"
        : "identity; I,Q,U,V=(I_camera,Q_camera,U_camera,V_camera)");
    if (trace) {
        text("stokes_convention", "camera_frame");
        text("polarization_basis", "camera_screen");
        text("evpa_0", evpa_0.c_str());
        text("trace_stokes_basis", "samples and final_propagated: transported internal screen (unchanged by evpa_0); final_observed: selected output N/W basis");
    } else {
        text("image_array_order", "y,x");
        text("image_display_origin", "lower");
    }
}
