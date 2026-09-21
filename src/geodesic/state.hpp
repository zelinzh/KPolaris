#pragma once

#include "common/vec.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct TransportState {
    Vec4<Real> x;
    // Camera rays store the physical future-directed photon wavevector in
    // both passes: d(lambda)<0 traces into the past, d(lambda)>0 toward the
    // observer. e1/e2 retain the same transported camera polarization frame.
    Vec4<Real> k;
    Vec4<Real> e1;
    Vec4<Real> e2;
};

template<class Real>
KPOLARIS_INLINE TransportState<Real> operator+(const TransportState<Real>& a,
                                            const TransportState<Real>& b) {
    return {a.x + b.x, a.k + b.k, a.e1 + b.e1, a.e2 + b.e2};
}

template<class Real>
KPOLARIS_INLINE TransportState<Real> operator*(const TransportState<Real>& a,
                                            Real s) {
    return {a.x * s, a.k * s, a.e1 * s, a.e2 * s};
}

template<class Real>
KPOLARIS_INLINE TransportState<Real> operator*(Real s,
                                            const TransportState<Real>& a) {
    return a * s;
}

template<class Real>
KPOLARIS_INLINE TransportState<Real> axpy(const TransportState<Real>& y,
                                       Real a,
                                       const TransportState<Real>& x) {
    return y + x * a;
}

template<class Real>
struct FrameErrors {
    Real k_null = Real(0);
    Real e1_dot_k = Real(0);
    Real e2_dot_k = Real(0);
    Real e1_norm = Real(0);
    Real e2_norm = Real(0);
    Real e1_dot_e2 = Real(0);
};

template<class Metric, class Real>
KPOLARIS_INLINE FrameErrors<Real> frame_errors(const Metric& metric,
                                            const TransportState<Real>& s) {
    FrameErrors<Real> err;
    err.k_null = metric.dot(s.x, s.k, s.k);
    err.e1_dot_k = metric.dot(s.x, s.e1, s.k);
    err.e2_dot_k = metric.dot(s.x, s.e2, s.k);
    err.e1_norm = metric.dot(s.x, s.e1, s.e1) - Real(1);
    err.e2_norm = metric.dot(s.x, s.e2, s.e2) - Real(1);
    err.e1_dot_e2 = metric.dot(s.x, s.e1, s.e2);
    return err;
}

template<class Real>
KPOLARIS_INLINE Real max_frame_error(const FrameErrors<Real>& err) {
    Real out = abs_val(err.k_null);
    out = max_val(out, abs_val(err.e1_dot_k));
    out = max_val(out, abs_val(err.e2_dot_k));
    out = max_val(out, abs_val(err.e1_norm));
    out = max_val(out, abs_val(err.e2_norm));
    out = max_val(out, abs_val(err.e1_dot_e2));
    return out;
}

} // namespace kpolaris
