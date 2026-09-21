#!/usr/bin/env python3
"""Plot matched full/thin-equatorial images, optionally with rotation disabled."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from plot_direct_only_compare import read
from kpolaris_image import load_image, image_extent


def match(reference, candidate):
    prefixes=('riaf_','iharm_','kharma_','athenak_','bhac_','hamr_','torus_')
    keys=('model','coordinate','camera','spin','radius','camera_radius','nx','ny',
          'inclination','inclination_rad','fov','fovy','xspan','yspan','x_offset','y_offset',
          'image_width_x_M','image_width_y_M','selected_frequency','evpa_0','polarization_basis',
          'stokes_convention','emission_type','emission_fit','slow_light',
          'slow_light_observation_time','slow_light_dump_list','slow_light_time_list',
          'inner_radius','outer_radius','direct_only','nonthermal_kappa','powerlaw_p',
          'powerlaw_eta','powerlaw_gamma_min','powerlaw_gamma_max','powerlaw_gamma_cutoff')
    for key in set(keys)|{k for k in reference if k.startswith(prefixes)}:
        if key not in reference and key not in candidate: continue
        if key not in reference or key not in candidate or not np.array_equal(reference[key],candidate[key]):
            raise ValueError('Mismatched physical metadata: '+key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('full',type=Path);p.add_argument('equatorial',type=Path)
    p.add_argument('--no-rotation',type=Path);p.add_argument('--frame',type=int,default=0)
    p.add_argument('--freq-index',type=int,default=0);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();paths=[args.full,args.equatorial]+([args.no_rotation] if args.no_rotation else [])
    products=[read(path,args.frame,args.freq_index) for path in paths]
    full,m0=products[0];eta=float(products[1][1].get('equatorial_h_over_r',0))
    if float(m0.get('equatorial_h_over_r',0)) not in (0,1) or not 0<eta<1:
        raise ValueError('Expected full emission, followed by a finite equatorial wedge')
    if int(m0.get('faraday_rotation',1))!=int(products[1][1].get('faraday_rotation',1)):
        raise ValueError('Full/thin pair must use the same propagation operator')
    if args.no_rotation and (int(products[2][1].get('faraday_rotation',1))!=0 or float(products[2][1].get('equatorial_h_over_r',0))!=eta):
        raise ValueError('Third image must have the same wedge and faraday_rotation=0')
    peak=float(full[0].max());flux=float(full[0].sum())
    if peak<=0 or flux<=0: raise ValueError('Full image has no positive emission')
    result={'frame':args.frame,'freq_index':args.freq_index,'equatorial_h_over_r':eta,'products':[],
            'note':'Same plasma normalization. Emission-only wedge; exterior absorption/conversion retained. Single snapshot, no beam convolution or registration.'}
    titles=['Full emission',f'Equatorial: h/r = {eta:g}']+(['Equatorial, rotation off'] if args.no_rotation else [])
    fig,axes=plt.subplots(2,len(products),figsize=(4*len(products),7.5),layout='constrained',squeeze=False)
    extent, _ = image_extent(load_image(args.full,args.frame,args.freq_index), 'M')
    for j,((stokes,meta),path,title) in enumerate(zip(products,paths,titles)):
        match(m0,meta)
        if stokes.shape!=full.shape: raise ValueError('Mismatched shape')
        intensity,q,u,v=stokes;total=stokes.sum(axis=(1,2));linear=np.hypot(q,u)
        if total[0]<=0: raise ValueError('Selected image has no emission')
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        row={'file':str(path.resolve()),'sha256':digest,'flux_over_full':float(total[0]/flux),
             'net_linear_fraction':float(np.hypot(total[1],total[2])/total[0]),
             'resolved_linear_fraction':float(linear.sum()/total[0]),
             'net_circular_fraction':float(total[3]/total[0]),
             'faraday_rotation':int(meta.get('faraday_rotation',1))}
        result['products'].append(row)
        im=axes[0,j].imshow(np.ma.masked_less_equal(intensity,0)/peak,origin='lower',extent=extent,
                            cmap=plt.get_cmap('magma').with_extremes(bad='black'),norm=LogNorm(1e-5,1))
        axes[0,j].set_title(title+'\n'+f'Flux / full = {row["flux_over_full"]:.4f}',fontsize=11)
        evpa=np.rad2deg(.5*np.arctan2(u,q))
        valid=(intensity>1e-3*intensity.max())&(linear>1e-3*linear.max())
        pol=axes[1,j].imshow(np.ma.array(evpa,mask=~valid),origin='lower',extent=extent,
                            cmap='twilight',norm=Normalize(-90,90))
        axes[1,j].set_title(f'EVPA (camera); net LP = {100*row["net_linear_fraction"]:.2f}%',fontsize=11)
        for ax in axes[:,j]:
            ax.set_facecolor('#ececec');ax.set_xlabel('Screen x [M]');ax.set_ylabel('Screen y [M]')
    fig.colorbar(im,ax=axes[0,:].tolist(),shrink=.85,label='I / peak full I (common logarithmic scale)')
    fig.colorbar(pol,ax=axes[1,:].tolist(),shrink=.85,label='EVPA [deg]',ticks=[-90,-45,0,45,90])
    negative=float(np.maximum(products[1][0][0]-full[0],0).sum()/flux)
    result['negative_full_minus_wedge_I_over_full_flux']=negative
    if negative>1e-3: raise ValueError('Substantial negative full-minus-wedge I; verify matched inputs and convergence')
    model=m0.get('model','KPolaris');model=model.decode() if isinstance(model,bytes) else str(model)
    fig.suptitle(f'{model.upper()} | {full.shape[-1]} x {full.shape[-2]} | Equatorial source selection',fontsize=15)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'): fig.savefig(args.output.with_suffix(suffix),dpi=180)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plt.close(fig);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
