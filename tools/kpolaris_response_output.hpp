#pragma once

#include <algorithm>
#include <stdexcept>
#include "kpolaris_model_image_output.hpp"
#include "diagnostics/response_config.hpp"

inline std::string response_partition_name(kpolaris::SourcePartition p) {
    const char* names[]={"none","radial","region","plasma_region","near_far","thetae","sigma","beta","ne_cgs","b_cgs"};
    return names[static_cast<int>(p)];
}

inline void write_response_analysis(H5::Group& analysis,
    const kpolaris::ResponseConfig<kpolaris::DefaultReal>& c,
    const std::vector<double>& data, const Options& opt) {
    if (!c.enabled()) return;
    const size_t npix=size_t(opt.nx)*opt.ny;
    if (data.size()!=size_t(4)*c.channels()*npix)
        throw std::runtime_error("physical response buffer does not match its declared shape");
    H5::Group group=analysis.createGroup("physical_response");
    write_h5_int(group,"schema_version",1);
    write_h5_string(group,"partition",response_partition_name(c.partition));
    write_h5_int(group,"bins",c.bins);
    write_h5_string(group,"stokes_basis","observer; same convention as parent image");
    write_h5_string(group,"source_semantics","tagged seed emission propagated through the full baseline transfer operator");
    write_h5_string(group,"partition_semantics","fixed baseline partition; each emitting sample belongs to exactly one bin");
    write_h5_scalar(group,"observer_theta_rad",c.observer_theta);
    write_h5_scalar(group,"observer_phi_rad",c.observer_phi);
    write_h5_scalar(group,"funnel_angle_rad",c.funnel_angle);
    write_h5_scalar(group,"disk_angle_rad",c.disk_angle);
    write_h5_scalar(group,"sigma_boundary",c.sigma_boundary);
    write_h5_scalar(group,"beta_boundary",c.beta_boundary);
    std::vector<double> edges(c.edges,c.edges+c.edge_count);
    if (!edges.empty()) write_h5_dataset_1d(group,"partition_edges",edges);
    if (c.partition==kpolaris::SourcePartition::region)
        write_h5_string(group,"bin_labels","funnel_near,funnel_far,sheath_near,sheath_far,disk_near,disk_far");
    if (c.partition==kpolaris::SourcePartition::plasma_region)
        write_h5_string(group,"bin_labels","high_sigma,low_sigma_low_beta,low_sigma_high_beta,unavailable");
    if (c.partition==kpolaris::SourcePartition::near_far)
        write_h5_string(group,"bin_labels","near,far");
    if (c.partition==kpolaris::SourcePartition::region || c.partition==kpolaris::SourcePartition::near_far)
        write_h5_string(group,"region_definition","geometric polar cones and sign of spherical-position dot observer direction at phi=0; labels do not imply unbound outflow or path ordering");
    if (c.partition==kpolaris::SourcePartition::radial)
        write_h5_string(group,"edge_convention","radial edge bins include underflow/overflow; interior boundaries assigned to upper bin");
    if (c.partition>=kpolaris::SourcePartition::thetae)
        write_h5_string(group,"edge_convention","bin 0 below first edge; interior [lower,upper); penultimate >= last edge; last unavailable");
    auto cube=[&](H5::Group& g,const std::string& name,int channel,int component) {
        std::vector<double> values(size_t(c.bins)*npix);
        for (int k=0;k<c.bins;++k) {
            const size_t begin=(size_t(4)*(channel+k)+component)*npix;
            std::copy_n(data.begin()+begin,npix,values.begin()+size_t(k)*npix);
        }
        write_h5_dataset_3d(g,name,values,c.bins,opt.nx,opt.ny);
    };
    H5::Group source=group.createGroup("source");
    const char* stokes[]={"I","Q","U","V"};
    for (int s=0;s<4;++s) cube(source,std::string(stokes[s])+"_inv",0,s);
    write_h5_int(group,"response_available",c.responses()?1:0);
    if (!c.responses()) return;
    const char* parameters[]={"none","density_scale","temperature_scale","magnetic_scale","coefficients"};
    write_h5_string(group,"parameter",parameters[static_cast<int>(c.parameter)]);
    write_h5_scalar(group,"log_parameter_step",c.step);
    write_h5_string(group,"derivative_coordinate","q=ln(scale), evaluated at scale=1");
    write_h5_string(group,"fixed_quantities","metric, geodesic/sample grid, velocity, magnetic direction, baseline radiating-domain mask, baseline partition");
    write_h5_string(group,"density_scale_definition","ne multiplied by exp(q), B by exp(q/2); Thetae, sigma, beta unchanged");
    write_h5_string(group,"temperature_scale_definition","Thetae multiplied by exp(q) after the baseline electron prescription and floor; gas dynamics unchanged");
    write_h5_string(group,"magnetic_scale_definition","B multiplied by exp(q), sigma by exp(2q), beta by exp(-2q); Thetae and field direction fixed; variable kappa reevaluated");
    write_h5_string(group,"coefficient_response_definition","independent logarithmic scales of emission, absorption, rotation, conversion blocks; summed derivative corresponds to scaling all coefficients");
    write_h5_string(group,"method","central differences of local discrete transfer-step blocks, propagated through the baseline homogeneous operator; O(h^2) derivative accuracy; four full perturbed transfers on the same sample grid");
    const char* mechanisms[]={"emission","absorption","rotation","conversion"};
    H5::Group response=group.createGroup("derivative");
    for (int m=0;m<4;++m) {
        H5::Group block=response.createGroup(mechanisms[m]);
        for (int s=0;s<4;++s) cube(block,std::string("d")+stokes[s]+"_inv_dlogp",c.response_channel(m,0),s);
    }
    H5::Group reruns=group.createGroup("reruns");
    const char* variants[]={"plus_h","minus_h","plus_half_h","minus_half_h"};
    for (int v=0;v<4;++v) {
        H5::Group g=reruns.createGroup(variants[v]);
        write_h5_scalar(g,"log_parameter_offset",c.step*(v%2==0?1:-1)*(v<2?1:0.5));
        for (int s=0;s<4;++s) {
            const size_t begin=(size_t(4)*c.rerun_channel(v)+s)*npix;
            std::vector<double> values(data.begin()+begin,data.begin()+begin+npix);
            write_h5_dataset_2d(g,std::string(stokes[s])+"_inv",values,opt.nx,opt.ny);
        }
    }
}
