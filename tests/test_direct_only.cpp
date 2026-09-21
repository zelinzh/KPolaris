#include <cmath>
#include <iostream>
#include <stdexcept>
#include "geodesic/pass_b.hpp"

using namespace kpolaris;
using Real=double;
void check(bool v,const char* text) { if(!v) throw std::runtime_error(text); }
void close(Real a,Real b,Real tol,const char* text) { check(std::isfinite(a)&&std::abs(a-b)<tol,text); }

// Manufactured constant-acceleration trajectory: z=2+s-s^2/2. This
// tests the event solver against an analytic root, independently of Kerr.
struct AcceleratedPath : KerrSchildInMetric<Real> {
    AcceleratedPath():KerrSchildInMetric<Real>(0,0) {}
    void connection(const Vec4<Real>&, Real g[4][4][4]) const {
        for(int m=0;m<4;++m)for(int a=0;a<4;++a)for(int b=0;b<4;++b)g[m][a][b]=0;
        g[3][0][0]=1;
    }
};
void analytic_events() {
    check(vertical_turn_bracket(2.,1.,2.,-1.,2.,5.),"upper vertical maximum");
    check(vertical_turn_bracket(-2.,-1.,-2.,1.,2.,5.),"lower vertical minimum");
    check(vertical_turn_bracket(2.,-1.,2.,1.,-2.,5.),"signed affine step");
    check(!vertical_turn_bracket(1.,-1.,-1.,-1.,2.,5.),"equatorial crossing is not a turn");
    check(!vertical_turn_bracket(2.,-1.,2.,1.,2.,5.),"wrong direction reversal");
    check(!vertical_turn_bracket(0.,0.,0.,0.,2.,5.),"equatorial ray");
    AcceleratedPath metric; TransportState<Real> first;
    first.x=Vec4<Real>(0,4,0,2);first.k=Vec4<Real>(1,0,0,1);
    for(int composed : {0,1}) {
        Real h=2;auto last=metric_surface_path_state(metric,first,h,composed);
        check(truncate_at_first_vertical_turn(metric,first,last,h,composed),"analytic turn detected");
        close(h,1,2e-14,"analytic event affine position");
        close(last.x[3],2.5,2e-14,"analytic event height");
        close(last.k[3],0,2e-14,"analytic event derivative");
    }
}
void coordinate_derivatives() {
    const Real r=7,th=1.1,dr=-.6,dth=.08;
    TransportState<Real> s;s.x=Vec4<Real>(0,r,th,.4);s.k=Vec4<Real>(1,dr,dth,.03);
    Real z,v; KerrSchildSphericalMetric<Real> ks(1,.8);KerrBoyerLindquistMetric<Real> bl(1,.8);
    const Real expected=std::cos(th)*dr-r*std::sin(th)*dth;
    vertical_position_velocity(ks,s,z,v);close(v,expected,1e-14,"spherical dz/ds");
    vertical_position_velocity(bl,s,z,v);close(v,expected,1e-14,"BL dz/ds");
    KerrFMKSMetric<Real> fmks(1,.8,0,.3,.5,14,.82,.75782);
    s.x[1]=std::log(r);s.x[2]=fmks.native_x2_from_theta(s.x[1],th);
    s.k[1]=dr/r;
    s.k[2]=(dth-fmks.dtheta_dx1(s.x[1],s.x[2])*s.k[1])/fmks.dtheta_dx2(s.x[1],s.x[2]);
    vertical_position_velocity(fmks,s,z,v);close(z,r*std::cos(th),1e-12,"FMKS height");
    close(v,expected,1e-12,"FMKS radial-polar cross derivative");
    const Real eps=1e-5;
    const auto height=[&](Real step){auto x=s.x+s.k*step;return std::exp(x[1])*std::cos(fmks.theta_from_x2(x[1],x[2]));};
    close(v,(height(eps)-height(-eps))/(2*eps),1e-9,"independent coordinate finite difference");
    KerrSchildInMetric<Real> cart(1,.8);s.x[3]=z;s.k[3]=expected;
    vertical_position_velocity(cart,s,z,v);close(v,expected,1e-14,"Cartesian dz/ds");
}

struct GatedPlasma {
    Real freq_cgs=230e9;
    Real cutoff=-1e30;
    Real dlambda_scale() const{return 1;}
    template<class Metric>
    TransferCoeffs<Real> coefficients(const Metric& metric,const TransportState<Real>& s,Real) const {
        TransferCoeffs<Real> c;
        const Real r=radial_coordinate(metric,s.x),w=std::exp(-r*r/100);
        // Propagation is retained everywhere, including behind the cutoff.
        c.aI=.02*w;c.aQ=.005*w;c.rV=.2*w;c.rQ=.1*w;
        if(s.x[0]>=cutoff){c.jI=w;c.jQ=.3*w;c.jU=.1*w;c.jV=.02*w;}
        return c;
    }
};
void multifrequency_domain() {
    // The manufactured plasma extends beyond outer_radius. Every frequency
    // path must apply the requested domain, even when the model emits there.
    KerrSchildInMetric<Real> metric(1,.5);
    PassAParams<Real> p;p.spin=.5;p.camera.nx=6;p.camera.ny=6;
    p.camera.radius=50;p.camera.inclination=.3;p.camera.fov=.4;
    p.inner_radius=2;p.outer_radius=25;p.adaptive=1;p.adaptive_tolerance=1e-12;
    p.min_step=1e-10;p.max_step=.25;p.max_radiation_step=.05;p.max_steps=100000;
    GatedPlasma model;Real frequencies[1]={230e9};
    for(int direct : {0,1}) {
        p.direct_only=direct;
        const auto single=trace_pass_b_segment_model_pixel_metric(14,p,model,1,metric);
        Stokes<Real> fused[1],control[1];
        const auto a=trace_pass_b_segment_model_multifrequency_pixel_metric(14,p,model,1,frequencies,1,fused,metric);
        const auto b=trace_pass_b_segment_model_multifrequency_control_pixel_metric(14,p,model,1,frequencies,1,frequencies,1,control,metric);
        check(single.reason==TerminationReason::reached_camera&&a.reason==single.reason&&b.reason==single.reason,"frequency paths return");
        for(const auto& s : {fused[0],control[0]}) {
            const auto& ref=single.observed_stokes;
            const Real error=std::abs(s.I-ref.I)+std::abs(s.Q-ref.Q)+std::abs(s.U-ref.U)+std::abs(s.V-ref.V);
            check(error<1e-10*ref.I,"single and fused/control frequencies enforce the same radiation domain");
        }
    }
}
void kerr_and_emission_gate() {
    KerrSchildInMetric<Real> metric(1,.9375);
    PassAParams<Real> p;p.camera.nx=6;p.camera.ny=6;p.camera.radius=50;
    p.camera.fov=.4;p.inner_radius=1.4;p.outer_radius=30;
    p.spin=.9375;p.adaptive=1;p.adaptive_tolerance=1e-12;p.min_step=1e-10;
    p.max_step=.25;p.max_steps=200000;p.direct_only=1;
    int turns=0,no_turns=0;
    for(Real inclination : {17.,85.,163.}) {
        p.camera.inclination=inclination*3.14159265358979323846/180;
        for(int pixel=0;pixel<36;++pixel) {
            const auto endpoint=trace_pass_a_segment_endpoint_pixel_metric(pixel,p,metric);
            check(endpoint.valid,"Kerr endpoint valid");
            auto full_p=p;full_p.direct_only=0;
            if(endpoint.reason==TerminationReason::reached_direct_turn) {
                ++turns;Real z,v;vertical_position_velocity(metric,endpoint.state,z,v);
                close(v,0,1e-10,"refined Kerr dz/ds");
                auto after=rk4_step(metric,endpoint.state,.001);Real z1,v1;vertical_position_velocity(metric,after,z1,v1);
                check(z1*v1<0,"after first turn ray returns toward midplane");
                if(turns==1) {
                    GatedPlasma model;
                    p.max_radiation_step=.002;full_p.max_radiation_step=.002;
                    const auto direct=trace_pass_b_segment_model_pixel_metric(pixel,p,model,1,metric);
                    model.cutoff=endpoint.state.x[0];
                    const auto gate=trace_pass_b_segment_model_pixel_metric(pixel,full_p,model,1,metric);
                    check(direct.reason==TerminationReason::reached_camera&&gate.reason==TerminationReason::reached_camera,"both transfer calculations return");
                    const auto a=direct.observed_stokes,b=gate.observed_stokes;
                    const Real error=(std::abs(a.I-b.I)+std::abs(a.Q-b.Q)+std::abs(a.U-b.U)+std::abs(a.V-b.V))/a.I;
                    std::cout<<"Full-path emission gate vs endpoint relative Stokes L1: "<<error<<'\n';
                    check(error<1e-3,"direct-only agrees with independent full-path j gate");
                    p.max_radiation_step=1;
                }
            } else {
                ++no_turns;const auto full=trace_pass_a_segment_endpoint_pixel_metric(pixel,full_p,metric);
                check(endpoint.reason==full.reason&&endpoint.steps==full.steps,"no-turn path unchanged");
                for(int mu=0;mu<4;++mu) check(endpoint.state.x[mu]==full.state.x[mu],"no-turn endpoint bitwise unchanged");
            }
        }
    }
    check(turns>0&&no_turns>0,"both turning and nonturning rays exercised");
    std::cout<<"Kerr rays: "<<turns<<" turns, "<<no_turns<<" unchanged\n";
}
int main(int argc,char**argv) {
    Kokkos::initialize(argc,argv);int status=0;
    try{analytic_events();coordinate_derivatives();multifrequency_domain();kerr_and_emission_gate();std::cout<<"Direct-only tests passed\n";}
    catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
    Kokkos::finalize();return status;
}
