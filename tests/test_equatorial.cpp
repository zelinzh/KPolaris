#include <cmath>
#include <iostream>
#include <stdexcept>
#include "geodesic/pass_b.hpp"

using namespace kpolaris;
using Real = double;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void close(Real a, Real b, Real tol, const char* message) {
    check(std::isfinite(a) && std::abs(a-b) <= tol, message);
}

// Manufactured q=z/r=z trajectory with an analytic source column and foreground
// screen. This is an integration fixture, not a physical Kerr metric.
struct LinearHeight : KerrSchildInMetric<Real> {
    LinearHeight() : KerrSchildInMetric<Real>(0,0) {}
    Real radial_coordinate(const Vec4<Real>&) const { return 1; }
    void connection(const Vec4<Real>&, Real g[4][4][4]) const {
        for (int m=0;m<4;++m) for (int a=0;a<4;++a) for (int b=0;b<4;++b) g[m][a][b]=0;
    }
};
void analytic_slab() {
    LinearHeight metric;
    for (Real eta : {.01, .00001}) for (int samples : {2,8,16})
    for (int composed : {0,1}) for (Real direction : {-1.,1.}) {
        EmissionSelection<Real> selection; selection.equatorial_h_over_r=eta; selection.equatorial_samples=samples;
        TransportState<Real> state; state.x=Vec4<Real>(0,1,0,-10*eta*direction); state.k=Vec4<Real>(1,0,0,direction);
        Stokes<Real> stokes;
        const Real length=20*eta, absorption=1/eta;
        Real distance=0; int steps=0;
        while (distance<length*(1-1e-13)) {
            Real h=std::min(13.73*eta,length-distance);
            auto last=emission_selection_path_state(metric,state,h,composed);
            auto mid=rk4_step(metric,state,h*.5);
            const int n=equatorial_resolution_substeps(selection,metric,state,mid,last,2.);
            h/=n; last=emission_selection_path_state(metric,state,h,composed); mid=rk4_step(metric,state,h*.5);
            refine_equatorial_boundary(selection,metric,state,mid,last,h,composed,2.);
            TransferCoeffs<Real> c;c.jI=1;c.aI=absorption;
            apply_emission_selection(selection,metric,mid,c);
            semi_analytic_stokes_step(stokes,c,h);
            state=last;distance+=h;
            check(++steps<10000,"thin slab makes forward progress");
        }
        const Real expected=(1-std::exp(-2*eta*absorption))/absorption*std::exp(-9*eta*absorption);
        close(stokes.I/expected,1,2e-10,"analytic emitting slab and exterior absorption");
    }
    // Exactly at the midpoint of a trial step: an equality must also split.
    EmissionSelection<Real> sel;sel.equatorial_h_over_r=.01;
    TransportState<Real> first;first.x=Vec4<Real>(0,1,0,-.03);first.k=Vec4<Real>(1,0,0,1);
    Real h=.04;auto mid=rk4_step(metric,first,h*.5),last=rk4_step(metric,first,h);
    check(refine_equatorial_boundary(sel,metric,first,mid,last,h,0,2.),"midpoint boundary detected");
    close(h,.02,1e-14,"midpoint boundary location");
}
void coordinates_and_coefficients() {
    const Real r=7,theta=1.1,a=.8;
    KerrBoyerLindquistMetric<Real> bl(1,a);KerrSchildSphericalMetric<Real> sks(1,a);
    KerrFMKSMetric<Real> fmks(1,a,0,.3,.5,14,.82,.75782);KerrSchildInMetric<Real> cart(1,a);
    Vec4<Real> x(0,r,theta,.4);
    close(equatorial_height_ratio(bl,x),std::cos(theta),1e-14,"BL physical angle");
    close(equatorial_height_ratio(sks,x),std::cos(theta),1e-14,"SKS physical angle");
    x[1]=std::log(r);x[2]=fmks.native_x2_from_theta(x[1],theta);
    close(equatorial_height_ratio(fmks,x),std::cos(theta),1e-12,"FMKS uses physical theta");
    x=Vec4<Real>(0,std::sqrt(r*r+a*a)*std::sin(theta),0,r*std::cos(theta));
    close(equatorial_height_ratio(cart,x),std::cos(theta),1e-14,"Cartesian Kerr radius, not cylindrical radius");
    TransferCoeffs<Real> base;base.jI=1;base.jQ=.2;base.jU=.3;base.jV=.1;
    base.aI=2;base.aQ=.02;base.aU=.03;base.aV=.01;base.rQ=.4;base.rU=.5;base.rV=.6;
    TransportState<Real> state;state.x=Vec4<Real>(0,r,theta,0);
    EmissionSelection<Real> sel;sel.equatorial_h_over_r=.01;
    auto c=base;apply_emission_selection(sel,bl,state,c);
    check(c.jI==0&&c.jQ==0&&c.jU==0&&c.jV==0,"all exterior emission removed");
    check(c.aI==base.aI&&c.aQ==base.aQ&&c.aU==base.aU&&c.aV==base.aV,"exterior dichroic absorption retained");
    check(c.rQ==base.rQ&&c.rU==base.rU&&c.rV==base.rV,"exterior Faraday transfer retained");
    sel.faraday_rotation=0;c=base;apply_emission_selection(sel,bl,state,c);
    check(c.rV==0&&c.rQ==base.rQ&&c.rU==base.rU,"rotation switch retains conversion");
    for (Real width : {0.,1.}) {
        sel.equatorial_h_over_r=width;sel.faraday_rotation=1;c=base;apply_emission_selection(sel,bl,state,c);
        check(c.jI==base.jI&&c.jQ==base.jQ&&c.rV==base.rV,"disabled or full-sphere selection unchanged");
        check(equatorial_resolution_substeps(sel,bl,state,state,state,100.)==1,"no extra sampling when disabled");
    }
}
int main(int argc, char** argv) {
    Kokkos::initialize(argc,argv);int status=0;
    try { analytic_slab();coordinates_and_coefficients();std::cout<<"Equatorial source tests passed\n"; }
    catch (const std::exception& e) { std::cerr<<e.what()<<'\n';status=1; }
    Kokkos::finalize();return status;
}
