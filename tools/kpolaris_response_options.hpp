#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "kpolaris_model_image_options.hpp"
#include "diagnostics/response_config.hpp"

inline kpolaris::ResponseConfig<ImageReal> make_response_config(
    const Options& opt, int radial_bins, ImageReal radial_min, ImageReal radial_max) {
    using P = kpolaris::PlasmaParameter;
    using B = kpolaris::SourcePartition;
    kpolaris::ResponseConfig<ImageReal> c;
    if (opt.analysis_response == "none") c.parameter=P::none;
    else if (opt.analysis_response == "density_scale") c.parameter=P::density_scale;
    else if (opt.analysis_response == "temperature_scale") c.parameter=P::temperature_scale;
    else if (opt.analysis_response == "magnetic_scale") c.parameter=P::magnetic_scale;
    else if (opt.analysis_response == "coefficients") c.parameter=P::coefficients;
    else throw std::runtime_error("analysis_response must be none, density_scale, temperature_scale, magnetic_scale, or coefficients");
    std::string partition=opt.analysis_partition;
    if (partition == "none" && c.responses()) partition="radial";
    if (partition == "none") c.partition=B::none;
    else if (partition == "radial") c.partition=B::radial;
    else if (partition == "region") c.partition=B::region;
    else if (partition == "plasma_region") c.partition=B::plasma_region;
    else if (partition == "near_far") c.partition=B::near_far;
    else if (partition == "thetae") c.partition=B::thetae;
    else if (partition == "sigma") c.partition=B::sigma;
    else if (partition == "beta") c.partition=B::beta;
    else if (partition == "ne_cgs") c.partition=B::ne_cgs;
    else if (partition == "b_cgs") c.partition=B::b_cgs;
    else throw std::runtime_error("unknown analysis_partition");
    if (c.enabled() && !opt.analysis_mode) throw std::runtime_error("analysis_response/analysis_partition require analysis_mode=1");
    const bool hdf5 = opt.output_format == "hdf5" ||
        (opt.output_format == "auto" && opt.output.size() >= 3 && opt.output.substr(opt.output.size()-3) == ".h5");
    if (c.enabled() && !hdf5)
        throw std::runtime_error("physical response/source partition products require HDF5 output");
    c.step=opt.analysis_response_step;
    if (!std::isfinite(c.step) || c.step < 1e-6 || c.step > 0.1)
        throw std::runtime_error("analysis_response_step must be finite and in [1e-6,0.1]");
    const ImageReal rad=ImageReal(3.14159265358979323846/180.0);
    c.funnel_angle=opt.analysis_funnel_angle_deg*rad;
    c.disk_angle=opt.analysis_disk_angle_deg*rad;
    if (!std::isfinite(c.funnel_angle) || !std::isfinite(c.disk_angle) ||
        !(c.funnel_angle>0 && c.funnel_angle<c.disk_angle && c.disk_angle<ImageReal(0.5)*ImageReal(3.14159265358979323846)))
        throw std::runtime_error("analysis region angles must satisfy 0 < funnel < disk < 90 degrees");
    c.sigma_boundary=opt.analysis_sigma_boundary;
    c.beta_boundary=opt.analysis_beta_boundary;
    if (!std::isfinite(c.sigma_boundary) || !std::isfinite(c.beta_boundary) || c.sigma_boundary<=0 || c.beta_boundary<=0)
        throw std::runtime_error("analysis sigma/beta boundaries must be finite and positive");
    c.observer_theta=opt.inclination;
    if (c.partition==B::radial) {
        if (radial_bins+1 > kpolaris::max_partition_edges) throw std::runtime_error("too many response radial bins");
        c.bins=radial_bins; c.edge_count=radial_bins+1;
        for (int k=0;k<=radial_bins;++k) c.edges[k]=std::exp(std::log(radial_min)+ImageReal(k)/radial_bins*std::log(radial_max/radial_min));
    } else if (c.partition==B::region) c.bins=6;
    else if (c.partition==B::plasma_region) c.bins=4;
    else if (c.partition==B::near_far) c.bins=2;
    else if (c.partition!=B::none) {
        std::string edges=opt.analysis_partition_edges;
        if (edges.empty()) {
            if (c.partition==B::thetae) edges="1,3,10,30,100";
            if (c.partition==B::sigma || c.partition==B::beta) edges="0.01,0.1,1,10,100";
            if (c.partition==B::ne_cgs) edges="100,1000,10000,100000,1000000,10000000";
            if (c.partition==B::b_cgs) edges="0.1,1,10,100,1000";
        }
        std::istringstream stream(edges); std::string token;
        while (std::getline(stream,token,',')) {
            size_t used=0; const double x=std::stod(token,&used);
            if (used!=token.size() || !std::isfinite(x) || x<=0 ||
                c.edge_count>=kpolaris::max_partition_edges ||
                (c.edge_count && x<=c.edges[c.edge_count-1]))
                throw std::runtime_error("analysis_partition_edges must be finite, positive, strictly increasing (at most 61 edges)");
            c.edges[c.edge_count++]=ImageReal(x);
        }
        if (!c.edge_count || edges.back()==',') throw std::runtime_error("empty analysis partition edge");
        c.bins=c.edge_count+2; // underflow/overflow plus a separate unavailable bin
    }
    if (!opt.analysis_partition_edges.empty() && c.partition < B::thetae)
        throw std::runtime_error("analysis_partition_edges applies only to scalar plasma partitions");
    return c;
}
