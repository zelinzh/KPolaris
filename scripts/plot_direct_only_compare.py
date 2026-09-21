#!/usr/bin/env python3
"""Compare matching full/direct-only native images without aligning or renormalizing them."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, SymLogNorm


def read(path,frame,freq):
    with h5py.File(path) as h:
        g=h[f'frame_{frame}/freq_{freq}'];fg=h[f'frame_{frame}']
        if np.any(fg['diagnostics/reason'][...]!=1):raise ValueError('Incomplete image')
        data=np.stack([g[s+'_inv'][...] for s in 'IQUV'])
        if not np.isfinite(data).all():raise ValueError('Nonfinite Stokes')
        meta={k:h.attrs[k] for k in h.attrs}
        meta.update({k:v for k,v in g.attrs.items() if k.startswith(('riaf_','iharm_','kharma_','athenak_','bhac_','hamr_','torus_')) or k in ('emission_type','emission_fit','nonthermal_kappa','powerlaw_p','powerlaw_eta','powerlaw_gamma_min','powerlaw_gamma_max','powerlaw_gamma_cutoff','evpa_0','polarization_basis','stokes_convention')})
        meta.update({'selected_frequency':g.attrs.get('freq_hz',g.attrs.get('frequency_hz',g.attrs.get('freq',None)))})
        return data,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('full',type=Path);p.add_argument('direct',type=Path)
    p.add_argument('--frame',type=int,default=0);p.add_argument('--freq-index',type=int,default=0)
    p.add_argument('--output',type=Path,required=True,help='Figure output stem; PNG, PDF, JSON are written')
    a=p.parse_args();full,m0=read(a.full,a.frame,a.freq_index);direct,m1=read(a.direct,a.frame,a.freq_index)
    if int(m0.get('direct_only',0))!=0 or int(m1.get('direct_only',0))!=1:raise ValueError('Expected all, then direct_only=1')
    if full.shape!=direct.shape:raise ValueError('Mismatched image shapes')
    # Validate common physical, camera, and slow-light metadata while allowing
    # different execution settings. No image registration or flux rescaling.
    prefixes=('riaf_','iharm_','kharma_','athenak_','bhac_','hamr_','torus_')
    keys=('emission_type','emission_fit','nonthermal_kappa','powerlaw_p','powerlaw_eta','powerlaw_gamma_min','powerlaw_gamma_max','powerlaw_gamma_cutoff','evpa_0','polarization_basis','stokes_convention','model','coordinate','camera','spin','radius','camera_radius','image_width_x_M','image_width_y_M','inclination','inclination_rad','fov','fovy','nx','ny','freq','freq_hz','nfreq','selected_frequency',
          'slow_light','slow_light_observation_time','slow_light_dump_list','slow_light_time_list','outer_radius','inner_radius',
          'xspan','yspan','x_offset','y_offset','equatorial_h_over_r','faraday_rotation')
    for meta in (m0,m1):
        meta.setdefault('equatorial_h_over_r',0);meta.setdefault('faraday_rotation',1)
    for key in set(keys)|{k for k in m0 if k.startswith(prefixes)}:
        if key not in m0 and key not in m1:continue
        if key not in m0 or key not in m1 or not np.array_equal(m0[key],m1[key]):raise ValueError('Mismatched metadata: '+key)
    residual=full-direct;flux=full[0].sum();peak=full[0].max()
    if not flux>0:raise ValueError('Nonpositive full intensity')
    negative=float(np.maximum(-residual[0],0).sum()/flux)
    if negative>1e-3:raise ValueError('Full-minus-direct contains substantial negative I; check matched inputs and convergence')
    result={'full':str(a.full.resolve()),'direct':str(a.direct.resolve()),'frame':a.frame,'freq_index':a.freq_index,
            'direct_flux_fraction':float(direct[0].sum()/flux),'removed_flux_fraction':float(residual[0].sum()/flux),
            'negative_residual_I_over_full_flux':negative,
            'residual_stokes_over_full_I':(residual.sum(axis=(1,2))/flux).tolist(),
            'note':'Independent adaptive-transfer grids; difference is not an exact native source tag or a comparison to ipole.'}
    for key,path in [('full',a.full),('direct',a.direct)]:
        with path.open('rb') as f:result[key+'_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
    for ax,data,title in zip(axes[:2],(full[0],direct[0]),('All emission',f'Direct only ({100*result["direct_flux_fraction"]:.1f}% of flux)')):
        im=ax.imshow(data/peak,origin='lower',cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=1));ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes[:2],shrink=.7,label=r'$I/I_{\rm peak,all}$ (square-root scale)')
    v=max(float(np.abs(residual[0]).max()/peak),1e-12)
    im=axes[2].imshow(residual[0]/peak,origin='lower',cmap='RdBu_r',norm=SymLogNorm(v*.02,vmin=-v,vmax=v))
    axes[2].set_title('All minus direct');axes[2].set_xticks([]);axes[2].set_yticks([])
    fig.colorbar(im,ax=axes[2],shrink=.7,label=r'$\Delta I/I_{\rm peak,all}$ (symmetric log)')
    model=m0.get('model','KPolaris');model=model.decode() if isinstance(model,bytes) else str(model)
    fig.suptitle(f'{model.upper()} | First vertical-turn emission selection | {full.shape[-1]} x {full.shape[-2]}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(a.output.with_suffix('.'+ext),dpi=170)
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');plt.close(fig)
    print(json.dumps(result))

if __name__=='__main__':main()
