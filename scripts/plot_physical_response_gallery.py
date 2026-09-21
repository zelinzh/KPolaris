#!/usr/bin/env python3
"""Plot three complementary diagnostics from validated native response products."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from analyze_physical_responses import load_product, summarize, MECHANISMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--model',default='athenak')
    p.add_argument('--frame',type=int,default=0)
    p.add_argument('--freq-index',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    names=('density_scale','temperature_scale','magnetic_scale')
    products=[load_product(a.directory/f'{a.model}_{n}.h5',a.frame,a.freq_index) for n in names]
    checks=[summarize(*x) for x in products]
    if not all(c['validation_passed'] for c in checks): raise ValueError('Unvalidated response')
    selection_keys=('direct_only','equatorial_h_over_r','equatorial_samples','faraday_rotation')
    selections=[tuple(x[4]['emission_selection'][k] for k in selection_keys) for x in products]
    if any(x!=selections[0] for x in selections):
        raise ValueError('All parameter products must share emission selection and Faraday settings')
    stokes=products[0][0]; peak=stokes[0].max()
    if not all(np.allclose(x[0],stokes,rtol=1e-10,atol=peak*1e-12) for x in products):
        raise ValueError('All parameter products must share the same baseline image')
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    def project(d):
        pol=np.hypot(stokes[1],stokes[2])
        dp=np.divide(stokes[1]*d[1]+stokes[2]*d[2],pol,out=np.zeros_like(pol),where=pol>peak*1e-12)
        return np.stack((d[0],dp,d[3]))/peak
    def save(fig,name):
        for ext in ('png','pdf'): fig.savefig(a.output/f'{name}.{ext}',dpi=170)
        plt.close(fig)
    maps=np.stack([project(x[2].sum(axis=(0,2))) for x in products])
    fig,axes=plt.subplots(3,3,figsize=(10,8.3),layout='constrained')
    labels=('Density / mass unit','Electron temperature','Magnetic magnitude')
    for j,col in enumerate((r'$\partial I / \partial\ln p$',r'$\partial |Q+iU| / \partial\ln p$',r'$\partial V / \partial\ln p$')):
        vmax=max(np.abs(maps[:,j]).max(),1e-12)
        for i in range(3):
            ax=axes[i,j]; im=ax.imshow(maps[i,j],origin='lower',cmap='RdBu_r',norm=SymLogNorm(vmax*.02,vmin=-vmax,vmax=vmax))
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(col)
            if j==0:ax.set_ylabel(labels[i])
        fig.colorbar(im,ax=axes[:,j],shrink=.7,label=r'Response / $I_{\rm peak}$ (symmetric log)')
    fig.suptitle(f'{a.model.upper()} | Physical parameter responses\nRed: increase; blue: decrease. All transfer coefficients are recomputed.',fontsize=13)
    save(fig,'01_parameter_responses')
    # Radial location of a perturbation; signed responses to integrated observables.
    d=products[1][2]; meta=products[1][4]
    if meta['partition']!='radial':raise ValueError('Temperature product must use radial partition')
    edges=np.asarray(meta['partition_edges']); widths=np.diff(np.log(edges))
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    colors=('#b54b32','#326fa3','#8561aa','#27907a')
    keys=('dln_flux_I_dlogp','d_linear_fraction_dlogp','d_circular_fraction_dlogp')
    titles=(r'Flux response: $\partial\ln F_I/\partial\ln p$',r'Linear fraction: $\partial m_L/\partial\ln p$',r'Circular fraction: $\partial m_C/\partial\ln p$')
    for ax,key,title in zip(axes,keys,titles):
        total=np.zeros(len(widths))
        for m,c in zip(MECHANISMS,colors):
            y=np.array([b[key] for b in checks[1]['mechanisms'][m]['bins']])/widths
            total+=y;ax.stairs(y,edges,label=m.capitalize(),color=c,lw=1.6)
        ax.stairs(total,edges,color='black',lw=1.2,linestyle='--',label='Sum')
        ax.axhline(0,color='.65',lw=.7);ax.set_xscale('log');ax.set_xlabel(r'Perturbed radius $r\ [GM/c^2]$')
        ax.set_title(title);ax.set_ylabel(r'Response per $\Delta\ln r$');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,3))
    axes[0].legend(fontsize=8,ncol=2)
    fig.suptitle(f'{a.model.upper()} | Radial contributions to the electron-temperature response',fontsize=13)
    save(fig,'02_radial_mechanisms')
    tags=products[0][1];meta=products[0][4]
    if meta['partition']!='region':raise ValueError('Density product must use region partition')
    fig,axes=plt.subplots(2,3,figsize=(10,6.7),layout='constrained')
    for region,title in enumerate(('Polar cone','Sheath cone','Equatorial belt')):
        for side in range(2):
            k=2*region+side;ax=axes[side,region]
            im=ax.imshow(np.ma.masked_less_equal(tags[0,k]/peak,0),origin='lower',cmap='magma',norm=LogNorm(1e-6,1))
            ax.set_facecolor('#080719');ax.set_xticks([]);ax.set_yticks([])
            fraction=checks[0]['source_bins'][k]['stokes_over_total_I'][0]
            ax.set_title(f'{title}: {100*fraction:.3g}% of total flux')
            if region==0:ax.set_ylabel(('Near side','Far side')[side])
    fig.colorbar(im,ax=axes.ravel(),shrink=.75,label=r'Tagged observed $I / I_{\rm peak}$')
    fig.suptitle(f'{a.model.upper()} | Origin of observed emission\nGeometric cones: 0-20, 20-60, 60-90 degrees from the nearest spin pole',fontsize=13)
    save(fig,'03_source_regions')
    (a.output/'summary.json').write_text(json.dumps({'inputs':[str((a.directory/f'{a.model}_{n}.h5').resolve()) for n in names],
      'frame':a.frame,'freq_index':a.freq_index,
      'emission_selection':products[0][4]['emission_selection'],
      'image_shape':list(stokes.shape[1:]),'validation':[{'parameter':c['parameter'],'closure':c['source_closure_max_relative_l1'],'response':c['validation']} for c in checks],
      'note':'Functional diagnostic examples; not a resolution convergence study. Source plot floor: 1e-6 peak I. Cone labels are geometric, not dynamical jet identification.'},indent=2)+'\n')

if __name__=='__main__':main()
