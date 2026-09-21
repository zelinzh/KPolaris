#include <cmath>
#include <iostream>
#include <stdexcept>
#include <Kokkos_Core.hpp>
#include "diagnostics/response.hpp"

using Real=double;
using namespace kpolaris;

void close(const char* name, Real value, Real expected, Real tolerance=2e-6) {
    if (!std::isfinite(value) || std::abs(value-expected)>tolerance*std::max(Real(1),std::abs(expected)))
        throw std::runtime_error(std::string(name)+": got "+std::to_string(value)+", expected "+std::to_string(expected));
}

struct Ray {
    ResponseConfig<Real> config;
    Kokkos::View<Real*, Kokkos::HostSpace> data;
    Stokes<Real> s;
    Ray(int bins=2) {
        config.parameter=PlasmaParameter::coefficients;
        config.partition=SourcePartition::radial;
        config.bins=bins;
        config.step=1e-4;
        data=decltype(data)("response_test",4*config.channels());
    }
    void step(const TransferCoeffs<Real>& c, Real dl, int bin) {
        TransferCoeffs<Real> variants[4];
        for(int v=0;v<4;++v) variants[v]=response_scale_coefficients(c,config.step*(v%2?-1:1)*(v<2?1:0.5));
        response_transfer_step(config,data,0,1,bin,s,c,variants,dl);
        semi_analytic_stokes_step(s,c,dl);
    }
    Stokes<Real> source(int bin) const { return response_read<Real>(data,0,1,bin); }
    Stokes<Real> response(int m,int bin) const { return response_read<Real>(data,0,1,config.response_channel(m,bin)); }
    Stokes<Real> total() const {
        Stokes<Real> sum;
        for(int m=0;m<4;++m) for(int b=0;b<config.bins;++b) sum=response_add(sum,response(m,b));
        return sum;
    }
    Stokes<Real> difference(int first) const {
        return response_difference(response_read<Real>(data,0,1,config.rerun_channel(first)),
            response_read<Real>(data,0,1,config.rerun_channel(first+1)),first==0?2*config.step:config.step);
    }
};

void slab() {
    Ray ray;
    TransferCoeffs<Real> c; c.jI=2; c.aI=.7;
    for(int k=0;k<100;++k) ray.step(c,.02,k<40?0:1);
    const Real tau=1.4, source=2/.7;
    close("slab intensity",ray.s.I,source*(1-std::exp(-tau)),1e-12);
    close("source closure",ray.source(0).I+ray.source(1).I,ray.s.I,1e-12);
    close("joint emission/absorption response",ray.total().I,source*tau*std::exp(-tau));
    close("emission response",ray.response(0,0).I+ray.response(0,1).I,ray.s.I);
    close("absorption response",ray.response(1,0).I+ray.response(1,1).I,source*((tau+1)*std::exp(-tau)-1));
    close("full transfer FD",ray.total().I,ray.difference(0).I);
    close("half-step FD",ray.total().I,ray.difference(2).I);
}

void ordered_screens() {
    TransferCoeffs<Real> source; source.jI=2; source.jQ=1;
    TransferCoeffs<Real> screen; screen.rV=.8;
    Ray foreground; foreground.step(source,1,0); foreground.step(screen,1,1);
    close("screen rotation dQ",foreground.response(2,1).Q,-.8*std::sin(.8));
    close("screen rotation dU",foreground.response(2,1).U,.8*std::cos(.8));
    close("no screen emission",foreground.source(1).I,0,1e-15);
    Ray background; background.step(screen,1,1); background.step(source,1,0);
    close("background screen has no influence",background.response(2,1).U,0,1e-15);
    source.jQ=0; source.jU=1; screen.rV=0; screen.rQ=.6;
    Ray conversion; conversion.step(source,1,0); conversion.step(screen,1,1);
    close("conversion response",conversion.response(3,1).V,.6*std::cos(.6));
    close("seed V contribution includes later conversion",conversion.source(0).V,std::sin(.6),1e-12);
}

void mixed_transfer() {
    Ray ray(3);
    for(int k=0;k<75;++k) {
        TransferCoeffs<Real> c;
        c.jI=1+.2*std::sin(k*.2); c.jQ=.3; c.jU=-.1; c.jV=.05;
        c.aI=.4; c.aQ=.1; c.aU=.04; c.aV=-.02;
        c.rQ=.6*std::cos(k*.1); c.rU=-.3; c.rV=.8;
        ray.step(c,.03,k/25);
    }
    auto sum=response_add(response_add(ray.source(0),ray.source(1)),ray.source(2));
    close("mixed source I",sum.I,ray.s.I,1e-12); close("mixed source V",sum.V,ray.s.V,1e-12);
    auto a=ray.total(),b=ray.difference(2);
    close("mixed dI",a.I,b.I); close("mixed dQ",a.Q,b.Q);
    close("mixed dU",a.U,b.U); close("mixed dV",a.V,b.V);
}

void partitions_and_parameters() {
    AnalysisPlasmaDiagnostics<Real> p; p.valid=1; p.thetae=10;
    ResponseConfig<Real> c; c.partition=SourcePartition::thetae; c.edge_count=2;
    c.edges[0]=1; c.edges[1]=10; c.bins=4;
    close("upper edge ownership",response_partition_bin(c,6.,1.,1.,0.,p),2,0);
    p.valid=0; close("unknown bin",response_partition_bin(c,6.,1.,1.,0.,p),3,0);
    p.valid=1; c.partition=SourcePartition::sigma;
    close("RIAF sigma unavailable",response_partition_bin(c,6.,1.,1.,0.,p),3,0);
    c.partition=SourcePartition::region; c.bins=6; c.observer_theta=1.5707963267948966;
    close("near disk",response_partition_bin(c,6.,1.5707963267948966,1.,0.,p),4,0);
    close("far disk",response_partition_bin(c,6.,1.5707963267948966,-1.,0.,p),5,0);
    Real ne=3,thetae=5,b=7,sigma=.2,beta=4;
    PlasmaPerturbation<Real>{PlasmaParameter::density_scale,std::log(4.)}.apply(ne,thetae,b,sigma,beta);
    close("mass unit ne",ne,12,1e-12); close("mass unit B",b,14,1e-12);
    close("mass unit sigma",sigma,.2,1e-12); close("mass unit beta",beta,4,1e-12);
}

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int status=0;
    try { slab(); ordered_screens(); mixed_transfer(); partitions_and_parameters(); std::cout<<"Physical response tests passed\n"; }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; status=1; }
    Kokkos::finalize();
    return status;
}
