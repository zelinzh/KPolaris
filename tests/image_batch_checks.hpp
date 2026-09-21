#pragma once
#include <cmath>
#include <stdexcept>
#include "image/result.hpp"

inline void require_same_image(const ImageHostData& a,
                               const ImageHostData& b) {
    if (a.npix != b.npix || a.nfreq != b.nfreq) throw std::runtime_error("batch shape mismatch");
    if (a.image_i != b.image_i) throw std::runtime_error("batch image_i differs from serial dispatch");
    for (const auto value : b.image_i)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch image_i");
    if (a.image_q != b.image_q) throw std::runtime_error("batch image_q differs from serial dispatch");
    for (const auto value : b.image_q)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch image_q");
    if (a.image_u != b.image_u) throw std::runtime_error("batch image_u differs from serial dispatch");
    for (const auto value : b.image_u)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch image_u");
    if (a.image_v != b.image_v) throw std::runtime_error("batch image_v differs from serial dispatch");
    for (const auto value : b.image_v)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch image_v");
    if (a.closure_x != b.closure_x) throw std::runtime_error("batch closure_x differs from serial dispatch");
    for (const auto value : b.closure_x)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch closure_x");
    if (a.closure_k != b.closure_k) throw std::runtime_error("batch closure_k differs from serial dispatch");
    for (const auto value : b.closure_k)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch closure_k");
    if (a.final_null != b.final_null) throw std::runtime_error("batch final_null differs from serial dispatch");
    for (const auto value : b.final_null)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch final_null");
    if (a.frame_error != b.frame_error) throw std::runtime_error("batch frame_error differs from serial dispatch");
    for (const auto value : b.frame_error)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch frame_error");
    if (a.det_r != b.det_r) throw std::runtime_error("batch det_r differs from serial dispatch");
    for (const auto value : b.det_r)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch det_r");
    if (a.overlap_r11 != b.overlap_r11) throw std::runtime_error("batch overlap_r11 differs from serial dispatch");
    for (const auto value : b.overlap_r11)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch overlap_r11");
    if (a.overlap_r12 != b.overlap_r12) throw std::runtime_error("batch overlap_r12 differs from serial dispatch");
    for (const auto value : b.overlap_r12)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch overlap_r12");
    if (a.overlap_r21 != b.overlap_r21) throw std::runtime_error("batch overlap_r21 differs from serial dispatch");
    for (const auto value : b.overlap_r21)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch overlap_r21");
    if (a.overlap_r22 != b.overlap_r22) throw std::runtime_error("batch overlap_r22 differs from serial dispatch");
    for (const auto value : b.overlap_r22)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch overlap_r22");
    if (a.basis_identity_error != b.basis_identity_error) throw std::runtime_error("batch basis_identity_error differs from serial dispatch");
    for (const auto value : b.basis_identity_error)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch basis_identity_error");
    if (a.basis_rotation_angle != b.basis_rotation_angle) throw std::runtime_error("batch basis_rotation_angle differs from serial dispatch");
    for (const auto value : b.basis_rotation_angle)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite batch basis_rotation_angle");
    if (a.pass_a_steps != b.pass_a_steps) throw std::runtime_error("batch pass_a_steps differs from serial dispatch");
    if (a.steps != b.steps) throw std::runtime_error("batch steps differs from serial dispatch");
    if (a.reason != b.reason) throw std::runtime_error("batch reason differs from serial dispatch");
}
