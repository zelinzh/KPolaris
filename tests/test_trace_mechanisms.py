#!/usr/bin/env python3
"""Analytic mechanisms, path ordering, and convergence of CP production budgets."""
from pathlib import Path
import sys
import unittest
import tempfile
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_trace_mechanisms import replay, trace_arrays


class TraceMechanismTests(unittest.TestCase):
    def medium(self, n=1):
        return {'j':np.zeros((n,4)), 'alpha':np.zeros((n,4)),
                'rho':np.zeros((n,3)), 'dl':np.ones(n)}

    def test_intrinsic_emission_and_scalar_absorption(self):
        m=self.medium(); m['j'][0]=[2,0,0,.2]; m['alpha'][0,0]=1
        r=replay(**m)
        np.testing.assert_allclose(r['stokes'], [2*(1-np.exp(-1)),0,0,.2*(1-np.exp(-1))],atol=1e-14)
        self.assertAlmostEqual(r['channels'][0],r['stokes'][3])
        np.testing.assert_allclose(r['channels'][1:],0,atol=1e-14)

    def test_selective_absorption_creates_signed_V(self):
        m=self.medium(); m['alpha'][0]=[1,0,0,.5]
        r=replay(**m,initial=[1,0,0,0])
        expected=-np.exp(-1)*np.sinh(.5)
        self.assertAlmostEqual(r['stokes'][3],expected)
        self.assertAlmostEqual(r['channels'][1],expected)
        self.assertAlmostEqual(r['channels'][2],0)

    def test_conversion_and_back_conversion(self):
        m=self.medium(); m['rho'][0,0]=np.pi/2
        r=replay(**m,initial=[1,0,1,0])
        np.testing.assert_allclose(r['stokes'],[1,0,0,1],atol=1e-14)
        self.assertAlmostEqual(r['channels'][2],1)
        reverse=replay(**m,initial=[1,0,0,1])
        self.assertAlmostEqual(reverse['channels'][2],-1)
        self.assertAlmostEqual(reverse['channels'][3],1)
        self.assertAlmostEqual(reverse['stokes'][3],0)

    def test_large_rotation_depth_preserves_polarization(self):
        m=self.medium(); m['rho'][0,2]=1000
        r=replay(**m,initial=[1,1,0,0])
        self.assertAlmostEqual(np.hypot(r['stokes'][1],r['stokes'][2]),1)
        self.assertAlmostEqual(r['channels'][2],0)

    def test_rotation_then_conversion_is_not_reverse_order(self):
        m=self.medium(2); m['rho'][0,2]=np.pi/2; m['rho'][1,0]=np.pi/2
        r=replay(**m,initial=[1,1,0,0])
        self.assertAlmostEqual(r['stokes'][3],1)
        m['rho']=m['rho'][::-1]
        reverse=replay(**m,initial=[1,1,0,0])
        self.assertAlmostEqual(reverse['stokes'][3],0)
        disabled=replay(**m,initial=[1,1,0,0],conversion_scale=0)
        self.assertAlmostEqual(disabled['stokes'][3],0)

    def test_frequency_specific_terminal_state_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.h5'
            with h5py.File(path,'w') as h:
                h.attrs['nfreq']=2
                rays=h.create_group('rays')
                for name,value in (('sample_count',2),('pass_b_steps',2),('reason',1)):
                    rays.create_dataset(name,data=[value])
                rays.create_dataset('sample_offset',data=[0,2])
                shared=h.create_group('trace/shared')
                for name,value in (('r',[2.,3.]),('lambda',[0.,1.]),('dlambda',[1.,1.])):
                    shared.create_dataset(name,data=value)
                for fi in (0,1):
                    m=self.medium(2); m['j'][:]=[1,.1,0,.1*(fi+1)]
                    result=replay(**m)
                    g=h.create_group(f'trace/freq_{fi}'); g.attrs['frequency_hz']=(86e9,230e9)[fi]
                    for column,name in enumerate('IQUV'):
                        g.create_dataset('S'+name,data=result['history'][:,column])
                        g.create_dataset('j'+name,data=m['j'][:,column])
                        g.create_dataset('a'+name,data=m['alpha'][:,column])
                    for column,name in enumerate('QUV'):
                        g.create_dataset('rho'+name,data=m['rho'][:,column])
                    terminal=g.create_group('rays')
                    for column,name in enumerate('IQUV'):
                        terminal.create_dataset(f'final_propagated_{name}_inv',data=[result['stokes'][column]])
                    d=g.create_group('derived'); d.create_dataset('complete',data=[1]); d.create_dataset('intensity_freeze_valid',data=[0])
            arrays,recorded,final,radius,meta=trace_arrays(path,freq_index=1)
            self.assertAlmostEqual(final[3],.2)
            self.assertEqual(meta['frequency_hz'],230e9)
            with h5py.File(path,'a') as h: del h['trace/freq_1/rays']
            with self.assertRaisesRegex(ValueError,'terminal Stokes'):
                trace_arrays(path,freq_index=1)
            with h5py.File(path,'a') as h: h['trace/freq_0/derived/complete'][0]=0
            with self.assertRaisesRegex(ValueError,'unit-stride'):
                trace_arrays(path,freq_index=0)

    def test_production_attenuation_closure_and_refinement(self):
        m=self.medium(2)
        m['j'][:]=[1,.5,0,.02]; m['alpha'][:]=[.3,.1,.03,.01]
        m['rho'][:]=[.5,.1,1.2]
        r=replay(**m)
        self.assertAlmostEqual(sum(r['channels']),r['stokes'][3],places=13)
        np.testing.assert_allclose(r['observed_generation'].sum(axis=0),r['channels'][:3],atol=1e-14)
        # A separate RK4 integration of the unsplit constant-coefficient ODE.
        def rhs(s):
            return np.r_[1-.3*s[0]-np.dot(m['alpha'][0,1:],s[1:]),
                         m['j'][0,1:]-.3*s[1:]-m['alpha'][0,1:]*s[0]+np.cross(m['rho'][0],s[1:])]
        exact=np.zeros(4); dt=2/4000
        for _ in range(4000):
            a=rhs(exact); b=rhs(exact+dt*a/2); c=rhs(exact+dt*b/2); d=rhs(exact+dt*c)
            exact+=dt*(a+2*b+2*c+d)/6
        errors=[]
        for n in (1,2,4):
            x=replay(**m,substeps=n)
            errors.append(np.linalg.norm(x['stokes']-exact))
            self.assertAlmostEqual(sum(x['channels']),x['stokes'][3],places=13)
        self.assertGreater(errors[0]/errors[1],3.5)
        self.assertGreater(errors[1]/errors[2],3.5)


if __name__=='__main__': unittest.main()
