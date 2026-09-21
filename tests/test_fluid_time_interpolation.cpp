#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "image/slow_light_driver.hpp"
#include "model/grmhd_grid.hpp"
#include "grmhd/athenak_loader.hpp"
#include "geometry/kerr_fmks.hpp"
#include "geometry/kerr_schild_cartesian.hpp"

using Real = double;
using Grid = kpolaris::GRMHDGridRadiationModel<Real>;
using Direct = kpolaris::AthenaKDirectRadiationModel<Real>;
using Method = kpolaris::SlowLightInterpolation;

// Binary-exact values distinguish interpolating plasma from interpolating its
// nonlinear radiative response. The sigma cut is applied AFTER interpolation.
std::array<Real, 13> values(Real t) {
    const std::array<Real,13> a = {1, .125, .125, -.0625, .03125, .0625, .125, .03125,
                                  65536, 8, 16, .5, 2};
    const std::array<Real,13> b = {2, .5, -.0625, .125, .0625, -.03125, .0625, .125,
                                  131072, 32, 64, 1.5, 4};
    std::array<Real,13> out;
    for (int n=0;n<13;++n) out[n]=(1-t)*a[n]+t*b[n];
    return out;
}
Grid grid(Real t) {
    Grid m;
    m.n1=5; m.n2=6; m.n3=8; m.startx1=0; m.startx2=0; m.startx3=0;
    m.dx1=1; m.dx2=1./6; m.dx3=2*std::acos(-1.)/8; m.r_in=1; m.r_out=1000;
    m.poly_norm=.5*std::acos(-1.)/(1+1/(m.poly_alpha+1)/std::pow(m.poly_xt,m.poly_alpha));
    m.sigma_cut=1.25; m.has_derived_scalars=1;
    const int nc=m.n1*m.n2*m.n3;
    m.prims=Grid::RealView("p",8*nc); m.derived_scalars=Grid::RealView("d",5*nc);
    auto p=Kokkos::create_mirror_view(m.prims),d=Kokkos::create_mirror_view(m.derived_scalars);
    auto v=values(t);
    for (int i=0;i<m.n1;++i) for(int j=0;j<m.n2;++j) for(int k=0;k<m.n3;++k) {
        for(int n=0;n<8;++n) p(m.prim_index(n,i,j,k))=v[n];
        for(int n=0;n<5;++n) d(m.derived_index(n,i,j,k))=v[n+8];
    }
    Kokkos::deep_copy(m.prims,p); Kokkos::deep_copy(m.derived_scalars,d);
    return m;
}
Direct direct(Real t) {
    Direct m; m.nblocks=1; m.nx1=m.nx2=m.nx3=2; m.sigma_cut=1.25;
    m.index_nx=m.index_ny=m.index_nz=1;
    m.index_xmin=m.index_ymin=m.index_zmin=-20;
    m.index_xmax=m.index_ymax=m.index_zmax=20;
    m.index_inv_dx=m.index_inv_dy=m.index_inv_dz=1./40;
    m.extents=Direct::RealView("ext",6);m.cell_geom=Direct::RealView("geom",6);
    auto e=Kokkos::create_mirror_view(m.extents),g=Kokkos::create_mirror_view(m.cell_geom);
    for(int n=0;n<3;++n){e(2*n)=-20;e(2*n+1)=20;g(n)=-20;g(n+3)=.05;}
    Kokkos::deep_copy(m.extents,e);Kokkos::deep_copy(m.cell_geom,g);
    m.index_offsets=Direct::IntView("off",2);m.index_candidates=Direct::IntView("cand",1);
    auto o=Kokkos::create_mirror_view(m.index_offsets);o(0)=0;o(1)=1;
    Kokkos::deep_copy(m.index_offsets,o);Kokkos::deep_copy(m.index_candidates,0);
    m.prims=Direct::FloatView("p",64);m.derived=Direct::FloatView("d",40);
    auto p=Kokkos::create_mirror_view(m.prims),d=Kokkos::create_mirror_view(m.derived);
    auto v=values(t);const int native[8]={0,2,3,4,1,5,6,7};
    for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i){
        for(int n=0;n<8;++n)p(m.prim_index(0,n,i,j,k))=static_cast<float>(v[native[n]]);
        for(int n=0;n<5;++n)d(m.derived_index(0,n,i,j,k))=static_cast<float>(v[n+8]);
    }
    Kokkos::deep_copy(m.prims,p);Kokkos::deep_copy(m.derived,d);return m;
}
KPOLARIS_INLINE Real difference(const kpolaris::TransferCoeffs<Real>& a,
                               const kpolaris::TransferCoeffs<Real>& b) {
    const Real av[]={a.jI,a.jQ,a.jU,a.jV,a.aI,a.aQ,a.aU,a.aV,a.rQ,a.rU,a.rV};
    const Real bv[]={b.jI,b.jQ,b.jU,b.jV,b.aI,b.aQ,b.aU,b.aV,b.rQ,b.rU,b.rV};
    Real error=0;
    for(int n=0;n<11;++n)error=kpolaris::max_val(error,kpolaris::abs_val(av[n]-bv[n])/
        kpolaris::max_val(kpolaris::abs_val(bv[n]),Real(1e-100)));
    return error;
}
template<class Model,class Metric>
void check(Model a,Model b,Model middle,const Metric& metric,kpolaris::TransportState<Real> state) {
    using namespace kpolaris_image_detail;
    validate_slow_light_interpolation(a,b,Method::fluid);
    Kokkos::View<Real*> result("results",8);
    Kokkos::parallel_for("fluid_time_physics",1,KOKKOS_LAMBDA(int){
        SlowLightTemporalModel<Real,Model> temporal{a,b,10,12,11,a.freq_cgs,Method::fluid};
        const auto expected=middle.coefficients(metric,state,0);
        const auto actual=temporal.coefficients(metric,state,0);
        result(0)=difference(actual,expected);result(1)=actual.jI;
        Real rho,uu,ne,theta,B,sigma,beta;kpolaris::Vec4<Real> u,bvec;
        result(2)=temporal.fluid_state(metric,state,rho,uu,u,bvec,ne,theta,B,sigma,beta);
        result(3)=kpolaris::abs_val(metric.dot(state.x,u,u)+1);
        result(4)=kpolaris::abs_val(metric.dot(state.x,u,bvec));
        temporal.observation_time=10;result(5)=difference(temporal.coefficients(metric,state,0),a.coefficients(metric,state,0));
        temporal.observation_time=12;result(6)=difference(temporal.coefficients(metric,state,0),b.coefficients(metric,state,0));
        temporal.observation_time=11;temporal.interpolation=Method::coefficients;
        result(7)=difference(temporal.coefficients(metric,state,0),actual);
    });
    auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},result);
    if(r(0)>1e-10 || !(r(1)>0) || r(2)!=1 || r(3)>1e-11 || r(4)>1e-11 ||
       r(5)!=0 || r(6)!=0 || !(r(7)>1e-3)) throw std::runtime_error("fluid interpolation physics check failed");
    // A change in coordinates cannot silently mix unrelated primitive bases.
    b.spin+=.01;bool rejected=false;
    try{validate_slow_light_interpolation(a,b,Method::fluid);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("changed coordinate mapping accepted");
}
int main(int argc,char**argv){
    Kokkos::ScopeGuard guard(argc,argv);
    try{
        auto a=grid(0),b=grid(1),m=grid(.5);
        kpolaris::KerrFMKSMetric<Real> fmks(1,a.spin,a.startx1,a.hslope,a.mks_smooth,a.poly_alpha,a.poly_xt,a.poly_norm);
        kpolaris::TransportState<Real> s;
        s.x={0,2.1,.37,1.2};s.k={1,-.2,.03,.01};s.e1={0,.01,.7,.02};s.e2={0,.03,.02,.5};
        check(a,b,m,fmks,s);
        auto da=direct(0),db=direct(1),dm=direct(.5);
        kpolaris::KerrSchildInMetric<Real> cart(1,da.spin);
        s.x={0,8,2,1};s.k={1,-.2,.03,.01};s.e1={0,0,1,0};s.e2={0,0,0,1};
        check(da,db,dm,cart,s);
        db.index_xmax += 1;
        if (da.can_interpolate_fluid_with(db))
            throw std::runtime_error("changed meshblock domain accepted");
        m.precomputed_fluid_state=1;
        if(a.can_interpolate_fluid_with(m))throw std::runtime_error("cached four-vectors accepted");
        std::cout<<"PASS: FMKS and direct meshblock fluid interpolation, all 11 coefficients, endpoint limits, four-vector constraints, cutoff ordering, incompatible inputs\n";
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
