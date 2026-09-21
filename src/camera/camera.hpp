#pragma once

#include "geodesic/state.hpp"

namespace kpolaris {

enum class CameraModel : int {
    Pinhole = 0,
    ParallelPlane = 1,
};

template<class Real = DefaultReal>
struct CameraParams {
    int nx = 64;
    int ny = 64;
    Real radius = Real(100);
    Real inclination = Real(1.04719755119659774615);
    Real fov = Real(10);
    Real fov_y = Real(-1);
    Real xspan = Real(-1);
    Real yspan = Real(-1);
    Real x_offset = Real(0);
    Real y_offset = Real(0);
    CameraModel model = CameraModel::Pinhole;
};

template<class Real>
struct CameraBasis3 {
    Vec3<Real> position;
    Vec3<Real> forward;
    Vec3<Real> right;
    Vec3<Real> up;
};

template<class Real>
KPOLARIS_INLINE Vec3<Real> normalize3(Vec3<Real> v) {
    const Real n = norm(v);
    if (n <= Real(0)) {
        return Vec3<Real>(Real(0), Real(0), Real(0));
    }
    return v * (Real(1) / n);
}

template<class Real>
KPOLARIS_INLINE Vec3<Real> cross3(Vec3<Real> a, Vec3<Real> b) {
    return Vec3<Real>(a.y * b.z - a.z * b.y,
                      a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

template<class Real>
KPOLARIS_INLINE CameraBasis3<Real> make_parallel_plane_basis(const CameraParams<Real>& cam) {
    const Real si = Kokkos::sin(cam.inclination);
    const Real ci = Kokkos::cos(cam.inclination);

    CameraBasis3<Real> basis;
    basis.position = Vec3<Real>(cam.radius * si, Real(0), cam.radius * ci);
    basis.forward = normalize3(Vec3<Real>(-basis.position.x,
                                          -basis.position.y,
                                          -basis.position.z));
    basis.up = normalize3(Vec3<Real>(-ci, Real(0), si));
    // The camera stores the physical screen basis for the observed outgoing
    // photon direction k_out = -forward.  Using right = up x k_out keeps the
    // screen right-handed without any later camera-model-specific reflection.
    basis.right = normalize3(cross3(basis.up, basis.forward * Real(-1)));
    return basis;
}

template<class Real>
KPOLARIS_INLINE CameraBasis3<Real> make_pinhole_basis(const CameraParams<Real>& cam,
                                                   Real spin) {
    const Real si = Kokkos::sin(cam.inclination);
    const Real ci = Kokkos::cos(cam.inclination);

    CameraBasis3<Real> basis;
    // Cartesian Kerr-Schild embedding of the camera position at BL/KS phi = 0.
    basis.position = Vec3<Real>(cam.radius * si, spin * si, cam.radius * ci);
    basis.forward = normalize3(Vec3<Real>(-si, Real(0), -ci));
    basis.up = normalize3(Vec3<Real>(-cam.radius * ci, -spin * ci, cam.radius * si));
    // Image coordinates describe the source-side view, opposite the outgoing
    // photon screen vectors. In the flat, phi=0 limit +image x is +e_phi.
    // This Euclidean helper is not the finite-radius observer tetrad below.
    basis.right = normalize3(cross3(basis.up, basis.forward * Real(-1)));
    return basis;
}

template<class Real>
KPOLARIS_INLINE CameraBasis3<Real> make_camera_basis(const CameraParams<Real>& cam,
                                                  Real spin = Real(0)) {
    if (cam.model == CameraModel::Pinhole) {
        return make_pinhole_basis(cam, spin);
    }
    return make_parallel_plane_basis(cam);
}

template<class Real>
KPOLARIS_INLINE void pixel_offsets(int pixel, const CameraParams<Real>& cam, Real& sx, Real& sy) {
    const int i = pixel % cam.nx;
    const int j = pixel / cam.nx;
    const Real x_extent = cam.xspan > Real(0) ? Real(2) * cam.xspan : cam.fov;
    const Real y_extent = cam.yspan > Real(0) ? Real(2) * cam.yspan :
                          (cam.xspan > Real(0) ? Real(2) * cam.xspan * Real(cam.ny) / Real(cam.nx) :
                           (cam.fov_y > Real(0) ? cam.fov_y : cam.fov));
    sx = ((Real(i) + Real(0.5)) / Real(cam.nx) - Real(0.5)) * x_extent + cam.x_offset;
    sy = ((Real(j) + Real(0.5)) / Real(cam.ny) - Real(0.5)) * y_extent + cam.y_offset;
}

template<class Real>
KPOLARIS_INLINE Vec3<Real> pixel_direction(int pixel, const CameraParams<Real>& cam,
                                          Real spin = Real(0)) {
    // Euclidean sourceward viewing direction. Use initialize_camera_ray for
    // actual curved-spacetime wavevectors and screen bases.
    const CameraBasis3<Real> basis = make_camera_basis(cam, spin);
    Real sx = Real(0), sy = Real(0);
    pixel_offsets(pixel, cam, sx, sy);
    if (cam.model == CameraModel::ParallelPlane) {
        return basis.forward;
    }
    return normalize3(basis.forward + basis.right * sx + basis.up * sy);
}

template<class Metric, class Real>
KPOLARIS_INLINE Real solve_future_null_k0(const Metric& metric,
                                       const Vec4<Real>& x,
                                       const Vec3<Real>& spatial_k) {
    const Real A = metric.gcov(0, 0, x);
    Real B = Real(0);
    Real C = Real(0);
    const Real ks[3] = {spatial_k.x, spatial_k.y, spatial_k.z};
    for (int i = 0; i < 3; ++i) {
        B += metric.gcov(0, i + 1, x) * ks[i];
        for (int j = 0; j < 3; ++j) {
            C += metric.gcov(i + 1, j + 1, x) * ks[i] * ks[j];
        }
    }
    const Real disc = max_val(B * B - A * C, Real(0));
    return (-B - Kokkos::sqrt(disc)) / A;
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> normal_observer_velocity(const Metric& metric,
                                                 const Vec4<Real>& x) {
    Vec4<Real> u;
    for (int mu = 0; mu < ndim; ++mu) {
        u[mu] = -metric.gcon(mu, 0, x);
    }
    const Real n2 = metric.dot(x, u, u);
    if (n2 < Real(0)) {
        u = u * (Real(1) / Kokkos::sqrt(-n2));
    }
    return u;
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> lower_vector_camera(const Metric& metric,
                                            const Vec4<Real>& x,
                                            const Vec4<Real>& v) {
    Vec4<Real> out;
    for (int mu = 0; mu < ndim; ++mu) {
        Real sum = Real(0);
        for (int nu = 0; nu < ndim; ++nu) {
            sum += metric.gcov(mu, nu, x) * v[nu];
        }
        out[mu] = sum;
    }
    return out;
}

template<class Metric, class Real>
KPOLARIS_INLINE void normalize_camera_frequency(const Metric& metric,
                                             const Vec4<Real>& x,
                                             Vec4<Real>& k) {
    const Vec4<Real> ucam = normal_observer_velocity(metric, x);
    const Vec4<Real> ucov = lower_vector_camera(metric, x, ucam);
    Real omega = Real(0);
    for (int mu = 0; mu < ndim; ++mu) {
        omega += ucov[mu] * k[mu];
    }
    omega = -omega;
    if (omega > tiny_positive<Real>()) {
        k = k * (Real(1) / omega);
    }
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> make_screen_vector(const Metric& metric,
                                           const Vec4<Real>& x,
                                           const Vec4<Real>& k,
                                           const Vec3<Real>& spatial) {
    Vec4<Real> e(Real(0), spatial.x, spatial.y, spatial.z);

    Real denom = Real(0);
    for (int nu = 0; nu < ndim; ++nu) {
        denom += metric.gcov(0, nu, x) * k[nu];
    }

    Real numer = Real(0);
    for (int mu = 1; mu < ndim; ++mu) {
        for (int nu = 0; nu < ndim; ++nu) {
            numer += metric.gcov(mu, nu, x) * e[mu] * k[nu];
        }
    }
    e[0] = -numer / denom;
    return e;
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> normalize_spacelike(const Metric& metric,
                                            const Vec4<Real>& x,
                                            Vec4<Real> e) {
    const Real n2 = metric.dot(x, e, e);
    if (n2 <= Real(0)) {
        return e;
    }
    return e * (Real(1) / Kokkos::sqrt(n2));
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> orthogonalize_spacelike(const Metric& metric,
                                                const Vec4<Real>& x,
                                                Vec4<Real> e,
                                                const Vec4<Real>& against) {
    const Real denom = metric.dot(x, against, against);
    if (abs_val(denom) > tiny_positive<Real>()) {
        e = e - against * (metric.dot(x, e, against) / denom);
    }
    return e;
}


template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> raise_covector_camera(const Metric& metric,
                                              const Vec4<Real>& x,
                                              const Vec4<Real>& cov) {
    Vec4<Real> out;
    for (int mu = 0; mu < ndim; ++mu) {
        Real sum = Real(0);
        for (int nu = 0; nu < ndim; ++nu) {
            sum += metric.gcon(mu, nu, x) * cov[nu];
        }
        out[mu] = sum;
    }
    return out;
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> project_out_camera(const Metric& metric,
                                           const Vec4<Real>& x,
                                           Vec4<Real> a,
                                           const Vec4<Real>& b) {
    const Real b2 = metric.dot(x, b, b);
    if (abs_val(b2) > tiny_positive<Real>()) {
        a = a - b * (metric.dot(x, a, b) / b2);
    }
    return a;
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> normalize_camera_basis_vector(const Metric& metric,
                                                      const Vec4<Real>& x,
                                                      Vec4<Real> v) {
    const Real n2 = metric.dot(x, v, v);
    const Real n = Kokkos::sqrt(max_val(abs_val(n2), tiny_positive<Real>()));
    return v * (Real(1) / n);
}

template<class Real>
KPOLARIS_INLINE int levi_civita4(int i, int j, int k, int l) {
    if (i == j || i == k || i == l || j == k || j == l || k == l) {
        return 0;
    }
    int a[4] = {i, j, k, l};
    int sign = 1;
    for (int m = 0; m < 4; ++m) {
        for (int n = m + 1; n < 4; ++n) {
            if (a[m] > a[n]) {
                sign = -sign;
            }
        }
    }
    return sign;
}

template<class Real>
KPOLARIS_INLINE Real tetrad_orientation(const Vec4<Real>& e0,
                                      const Vec4<Real>& e1,
                                      const Vec4<Real>& e2,
                                      const Vec4<Real>& e3) {
    Real out = Real(0);
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            for (int k = 0; k < ndim; ++k) {
                for (int l = 0; l < ndim; ++l) {
                    out += Real(levi_civita4<Real>(i, j, k, l)) *
                           e0[i] * e1[j] * e2[k] * e3[l];
                }
            }
        }
    }
    return out;
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> camera_position_phi0(const Metric& metric,
                                             Real radius,
                                             Real inclination) {
    if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
        return metric.camera_position_phi0(radius, inclination);
    } else if constexpr (Metric::coordinate_system != CoordinateSystem::CartesianKS) {
        return Vec4<Real>(Real(0), radius, inclination, Real(0));
    } else {
        const Real si = Kokkos::sin(inclination);
        const Real ci = Kokkos::cos(inclination);
        return Vec4<Real>(Real(0), radius * si, metric.spin * si, radius * ci);
    }
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> native_vector_phi0_to_metric(const Metric& metric,
                                                     Real v0,
                                                     Real vx1,
                                                     Real vx2,
                                                     Real vphi,
                                                     Real radius,
                                                     Real inclination) {
    if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
        return Vec4<Real>(v0, vx1, vx2, vphi);
    } else if constexpr (Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
        // ipole constructs its camera tetrad in native ingoing MKS
        // coordinates, where x1 = log(r), x2 = theta/pi, and the time
        // and azimuth coordinates are Kerr--Schild.  Transform that trial
        // vector to Boyer--Lindquist before Gram--Schmidt orthogonalization:
        //   dt_KS   = dt_BL   + 2 M r / Delta dr,
        //   dphi_KS = dphi_BL + a / Delta dr.
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real r = max_val(radius, tiny_positive<Real>());
        const Real delta = max_val(r * r - Real(2) * metric.mass * r +
                                       metric.spin * metric.spin,
                                   tiny_positive<Real>());
        const Real vr = r * vx1;
        return Vec4<Real>(v0 - Real(2) * metric.mass * r * vr / delta,
                          vr,
                          pi * vx2,
                          vphi - metric.spin * vr / delta);
    } else if constexpr (Metric::coordinate_system != CoordinateSystem::CartesianKS) {
        const Real pi = Real(3.141592653589793238462643383279502884);
        return Vec4<Real>(v0, radius * vx1, pi * vx2, vphi);
    } else {
        const Real si = Kokkos::sin(inclination);
        const Real ci = Kokkos::cos(inclination);
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real vr = radius * vx1;
        const Real vth = pi * vx2;
        return Vec4<Real>(v0,
                          vr * si + vth * radius * ci - vphi * metric.spin * si,
                          vth * metric.spin * ci + vphi * radius * si,
                          vr * ci - vth * radius * si);
    }
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> ipole_centering_covector(const Metric& metric,
                                                 const Vec4<Real>& x) {
    if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
        return Vec4<Real>(Real(1), Real(1), Real(0), Real(0));
    } else if constexpr (Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
        const Real r = max_val(x[1], tiny_positive<Real>());
        const Real delta = max_val(r * r - Real(2) * metric.mass * r +
                                       metric.spin * metric.spin,
                                   tiny_positive<Real>());
        // Native MKS covector (1, 1, 0, 0), transformed using x1=log(r)
        // and the ingoing-KS-to-BL time relation above.
        return Vec4<Real>(Real(1),
                          Real(1) / r + Real(2) * metric.mass * r / delta,
                          Real(0), Real(0));
    } else if constexpr (Metric::coordinate_system != CoordinateSystem::CartesianKS) {
        return Vec4<Real>(Real(1), Real(1) / max_val(x[1], tiny_positive<Real>()), Real(0), Real(0));
    } else {
        const Real xx = x[1];
        const Real yy = x[2];
        const Real zz = x[3];
        const Real a2 = metric.spin * metric.spin;
        const Real radius2 = xx * xx + yy * yy + zz * zz;
        const Real s = radius2 - a2;
        const Real disc = Kokkos::sqrt(max_val(s * s + Real(4) * a2 * zz * zz,
                                               tiny_positive<Real>()));
        const Real r2 = max_val(Real(0.5) * (s + disc), tiny_positive<Real>());
        const Real inv_2r2 = Real(0.5) / r2;
        const Real dlogr_dx = xx * (Real(1) + s / disc) * inv_2r2;
        const Real dlogr_dy = yy * (Real(1) + s / disc) * inv_2r2;
        const Real dlogr_dz = zz * (Real(1) + (s + Real(2) * a2) / disc) * inv_2r2;
        return Vec4<Real>(Real(1), dlogr_dx, dlogr_dy, dlogr_dz);
    }
}

template<class Metric, class Real>
KPOLARIS_INLINE Vec4<Real> ipole_camera_observer_velocity(const Metric& metric,
                                                       const Vec4<Real>& x) {
    if constexpr (Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
        const Real r = max_val(x[1], tiny_positive<Real>());
        const Real delta = max_val(r * r - Real(2) * metric.mass * r +
                                       metric.spin * metric.spin,
                                   tiny_positive<Real>());
        // ipole uses the normal observer of the native KS time slicing.  Its
        // covector is -dt_KS, which acquires a radial component in BL.
        const Vec4<Real> cov(Real(-1),
                            -Real(2) * metric.mass * r / delta,
                            Real(0), Real(0));
        return raise_covector_camera(metric, x, cov);
    } else {
        return normal_observer_velocity(metric, x);
    }
}

template<class Metric, class Real>
KPOLARIS_INLINE void make_ipole_camera_tetrad(const Metric& metric,
                                           const CameraParams<Real>& cam,
                                           const Vec4<Real>& x,
                                           Vec4<Real>& e0,
                                           Vec4<Real>& e1,
                                           Vec4<Real>& e2,
                                           Vec4<Real>& e3) {
    e0 = ipole_camera_observer_velocity(metric, x);
    const Vec4<Real> center_cov = ipole_centering_covector(metric, x);
    e3 = raise_covector_camera(metric, x, center_cov);
    e2 = native_vector_phi0_to_metric(metric, Real(0), Real(0), Real(1), Real(0),
                                      cam.radius, cam.inclination);
    e1 = native_vector_phi0_to_metric(metric, Real(1), Real(1), Real(1), Real(1),
                                      cam.radius, cam.inclination);

    e0 = normalize_camera_basis_vector(metric, x, e0);
    e3 = project_out_camera(metric, x, e3, e0);
    e3 = project_out_camera(metric, x, e3, e0);
    e3 = normalize_camera_basis_vector(metric, x, e3);

    e2 = project_out_camera(metric, x, e2, e0);
    e2 = project_out_camera(metric, x, e2, e3);
    e2 = project_out_camera(metric, x, e2, e0);
    e2 = project_out_camera(metric, x, e2, e3);
    e2 = normalize_camera_basis_vector(metric, x, e2);

    e1 = project_out_camera(metric, x, e1, e0);
    e1 = project_out_camera(metric, x, e1, e2);
    e1 = project_out_camera(metric, x, e1, e3);
    e1 = project_out_camera(metric, x, e1, e0);
    e1 = project_out_camera(metric, x, e1, e2);
    e1 = project_out_camera(metric, x, e1, e3);
    e1 = normalize_camera_basis_vector(metric, x, e1);

    if (tetrad_orientation(e0, e1, e2, e3) < Real(0)) {
        e1 = e1 * Real(-1);
    }
}

template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> initialize_parallel_plane_camera_ray(const Metric& metric,
                                                                       int pixel,
                                                                       const CameraParams<Real>& cam) {
    const CameraBasis3<Real> basis = make_parallel_plane_basis(cam);
    const Vec3<Real> k_out_spatial = basis.forward * Real(-1);

    Real sx = Real(0), sy = Real(0);
    pixel_offsets(pixel, cam, sx, sy);

    TransportState<Real> state;
    const Vec3<Real> position = basis.position + basis.right * sx + basis.up * sy;
    state.x = Vec4<Real>(Real(0), position.x, position.y, position.z);
    const Real k0 = solve_future_null_k0(metric, state.x, k_out_spatial);
    Vec4<Real> k_out(k0, k_out_spatial.x, k_out_spatial.y, k_out_spatial.z);
    normalize_camera_frequency(metric, state.x, k_out);
    // Keep the physical future-directed photon wavevector in both passes.
    // Pass A traces into the past with negative affine steps; Pass B uses
    // positive steps without changing k or the transported screen basis.
    state.k = k_out;

    state.e1 = make_screen_vector(metric, state.x, k_out, basis.right);
    state.e1 = normalize_spacelike(metric, state.x, state.e1);
    state.e2 = make_screen_vector(metric, state.x, k_out, basis.up);
    state.e2 = orthogonalize_spacelike(metric, state.x, state.e2, state.e1);
    state.e2 = normalize_spacelike(metric, state.x, state.e2);
    return state;
}

template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> initialize_pinhole_camera_ray(const Metric& metric,
                                                                int pixel,
                                                                const CameraParams<Real>& cam) {
    TransportState<Real> state;
    state.x = camera_position_phi0(metric, cam.radius, cam.inclination);

    Vec4<Real> e0, e1, e2, e3;
    make_ipole_camera_tetrad(metric, cam, state.x, e0, e1, e2, e3);

    Real sx = Real(0), sy = Real(0);
    pixel_offsets(pixel, cam, sx, sy);
    const Real spatial_norm = Kokkos::sqrt(Real(1) + sx * sx + sy * sy);
    const Real kt0 = Real(1);
    const Real kt1 = sx / spatial_norm;
    const Real kt2 = sy / spatial_norm;
    const Real kt3 = Real(1) / spatial_norm;
    const Vec4<Real> spatial_dir = e1 * kt1 + e2 * kt2 + e3 * kt3;
    const Vec4<Real> k_out = e0 * kt0 + spatial_dir;

    // As for the parallel camera, k remains the physical future-directed
    // wavevector. The signed affine step determines the tracing direction.
    state.k = k_out;
    state.e1 = e1 - spatial_dir * kt1;
    state.e1 = normalize_spacelike(metric, state.x, state.e1);
    state.e2 = e2 - spatial_dir * kt2;
    state.e2 = orthogonalize_spacelike(metric, state.x, state.e2, state.e1);
    state.e2 = normalize_spacelike(metric, state.x, state.e2);
    return state;
}

template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> initialize_camera_ray(const Metric& metric,
                                                        int pixel,
                                                        const CameraParams<Real>& cam) {
    if (cam.model == CameraModel::Pinhole) {
        return initialize_pinhole_camera_ray(metric, pixel, cam);
    }
    return initialize_parallel_plane_camera_ray(metric, pixel, cam);
}


template<class Metric, class Real>
KPOLARIS_INLINE void initialize_screen_frame_from_reference(const Metric& metric,
                                                         const Vec4<Real>& x,
                                                         const Vec4<Real>& k,
                                                         Vec4<Real>& e1,
                                                         Vec4<Real>& e2) {
    Vec3<Real> k_spatial(k[1], k[2], k[3]);
    Vec3<Real> ref(Real(0), Real(0), Real(1));
    const Real k_norm = norm(k_spatial);
    if (k_norm > Real(0)) {
        const Real alignment = abs_val(dot(normalize3(k_spatial), ref));
        if (alignment > Real(0.9)) {
            ref = Vec3<Real>(Real(0), Real(1), Real(0));
        }
    }

    Vec3<Real> side = cross(k_spatial, ref);
    if (norm(side) <= Real(1e-30)) {
        side = Vec3<Real>(Real(1), Real(0), Real(0));
    }
    side = normalize3(side);
    Vec3<Real> up = normalize3(cross(k_spatial, side));

    e1 = make_screen_vector(metric, x, k, side);
    e1 = normalize_spacelike(metric, x, e1);
    e2 = make_screen_vector(metric, x, k, up);
    e2 = orthogonalize_spacelike(metric, x, e2, e1);
    e2 = normalize_spacelike(metric, x, e2);
}

} // namespace kpolaris
