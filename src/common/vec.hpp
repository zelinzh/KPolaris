#pragma once

#include "common/math.hpp"

namespace kpolaris {

template<class Real>
struct Vec3 {
    Real x = Real(0);
    Real y = Real(0);
    Real z = Real(0);

    KPOLARIS_INLINE Vec3() {}
    KPOLARIS_INLINE Vec3(Real x_in, Real y_in, Real z_in)
        : x(x_in), y(y_in), z(z_in) {}
};

template<class Real>
struct Vec4 {
    Real v[ndim];

    KPOLARIS_INLINE Vec4() : v{Real(0), Real(0), Real(0), Real(0)} {}
    KPOLARIS_INLINE Vec4(Real v0, Real v1, Real v2, Real v3)
        : v{v0, v1, v2, v3} {}

    KPOLARIS_INLINE Real& operator[](int i) { return v[i]; }
    KPOLARIS_INLINE const Real& operator[](int i) const { return v[i]; }
};

template<class Real>
KPOLARIS_INLINE Vec3<Real> operator+(const Vec3<Real>& a, const Vec3<Real>& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template<class Real>
KPOLARIS_INLINE Vec3<Real> operator-(const Vec3<Real>& a, const Vec3<Real>& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template<class Real>
KPOLARIS_INLINE Vec3<Real> operator*(const Vec3<Real>& a, Real s) {
    return {a.x * s, a.y * s, a.z * s};
}

template<class Real>
KPOLARIS_INLINE Vec3<Real> operator*(Real s, const Vec3<Real>& a) {
    return a * s;
}

template<class Real>
KPOLARIS_INLINE Real dot(const Vec3<Real>& a, const Vec3<Real>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<class Real>
KPOLARIS_INLINE Vec3<Real> cross(const Vec3<Real>& a, const Vec3<Real>& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

template<class Real>
KPOLARIS_INLINE Real norm(const Vec3<Real>& a) {
    const Real scale = max_val(abs_val(a.x), max_val(abs_val(a.y), abs_val(a.z)));
    if (scale == Real(0)) {
        return Real(0);
    }
    const Vec3<Real> scaled(a.x / scale, a.y / scale, a.z / scale);
    return scale * Kokkos::sqrt(dot(scaled, scaled));
}


template<class Real>
KPOLARIS_INLINE Vec4<Real> operator+(const Vec4<Real>& a, const Vec4<Real>& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> operator-(const Vec4<Real>& a, const Vec4<Real>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> operator*(const Vec4<Real>& a, Real s) {
    return {a[0] * s, a[1] * s, a[2] * s, a[3] * s};
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> operator*(Real s, const Vec4<Real>& a) {
    return a * s;
}

template<class Real>
KPOLARIS_INLINE Real max_abs_component(const Vec4<Real>& a) {
    Real out = abs_val(a[0]);
    for (int i = 1; i < ndim; ++i) {
        out = max_val(out, abs_val(a[i]));
    }
    return out;
}



template<class Real>
KPOLARIS_INLINE Real spatial_radius(const Vec4<Real>& x) {
    return Kokkos::sqrt(x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
}

template<class Real>
KPOLARIS_INLINE Real cylindrical_radius(const Vec4<Real>& x) {
    return Kokkos::sqrt(x[1] * x[1] + x[2] * x[2]);
}

template<class Real>
KPOLARIS_INLINE Real dot_euclidean(const Vec4<Real>& a, const Vec4<Real>& b) {
    Real out = Real(0);
    for (int i = 0; i < ndim; ++i) {
        out += a[i] * b[i];
    }
    return out;
}

} // namespace kpolaris
