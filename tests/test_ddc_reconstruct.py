#!/usr/bin/env python3
"""Numerical and ownership contracts for bounded DDC reconstruction."""
import io,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kpolaris_ddc_reconstruct import Reconstruction, read_member


class Archive:
    def __init__(self, arrays):
        self.payloads={}
        for key,a in arrays.items():
            out=io.BytesIO();np.save(out,a,allow_pickle=False);self.payloads[key]=out.getvalue()
    def read(self,name): return self.payloads[name]
    def namelist(self): return list(self.payloads)


class ReconstructionTest(unittest.TestCase):
    def codec(self):
        return SimpleNamespace(
            frame_tau=lambda *args: .3125,
            dataset_bits=lambda m,d:m['bits'],
            archive_member_name=lambda d,i,s:s,
            np_load_member=lambda a,n:np.load(io.BytesIO(a.read(n)),allow_pickle=False),
            unpack_int4=lambda p,shape:np.column_stack(((p&15).astype(np.int16)-8,(p>>4).astype(np.int16)-8)).astype(np.int8).ravel()[:np.prod(shape)].reshape(shape),
            decode_dataset=lambda *args,**kwargs:'reference')

    def fast(self,codec):
        # The production selector intentionally rejects this synthetic codec;
        # exercise numerical operations independently of the revision gate.
        fast=Reconstruction(codec,'reference');fast.mode='numpy';return fast

    def test_quantization_exception_and_logarithmic_outputs(self):
        rng=np.random.default_rng(410)
        for bits in (4,8,16):
            for dataset in ('prims.rho','prims.u','prims.uvec','prims.B'):
                for scale_dtype in (np.float32,np.float64):
                    shape=(2,3,5);start=rng.uniform(-5,5,shape).astype('f4');end=rng.uniform(-5,5,shape).astype('f4')
                    q=rng.integers(-7,8,shape,dtype=np.int8 if bits<=8 else np.int16)
                    scale=np.array([.001,.007],dtype=scale_dtype).reshape(2,1,1)
                    indices=np.array([0,7,29],dtype='u4');values=np.array([-.3,0,.4],dtype='f4')
                    arrays={'scale':scale,'exception_indices':indices,'exception_values':values}
                    if bits==4:
                        enc=(q.ravel()+8).astype('u1');arrays.update(q4_packed=enc[::2]|(enc[1::2]<<4),q4_shape=np.array(shape,dtype='i8'))
                    else:arrays[f'q{bits}']=q
                    archive=Archive(arrays);raw_before=dict(archive.payloads);codec=self.codec();fast=self.fast(codec)
                    expected=(.6875*start+.3125*end).astype('f4')
                    residual=(q.astype('f4')*scale).astype('f4');residual.ravel()[indices.astype('i8')]=values
                    expected=expected+residual
                    if dataset in ('prims.rho','prims.u'):expected=np.exp(expected,dtype='f4')
                    actual=fast.decode(archive,{'bits':bits},dataset,1,start=start,end=end,start_transformed=start,end_transformed=end)
                    self.assertEqual(actual.tobytes(),expected.tobytes())
                    self.assertEqual(archive.payloads,raw_before)
                    first=actual.tobytes()
                    later=fast.decode(archive,{'bits':bits},dataset,2,start=end,end=start,start_transformed=end,end_transformed=start)
                    self.assertFalse(np.shares_memory(actual,later));self.assertEqual(actual.tobytes(),first)

    def test_views_fortran_endian_lifetime_and_truncation(self):
        codec=self.codec()
        for values in (np.asfortranarray(np.arange(24,dtype='>f4').reshape(2,3,4)),np.array(4,dtype='i8'),np.empty((0,3),dtype='f4')):
            archive=Archive({'v':values});result=read_member(codec,archive,'v')
            self.assertEqual(result.dtype,values.dtype);self.assertEqual(result.shape,values.shape)
            self.assertEqual(result.tobytes(),values.tobytes());self.assertFalse(result.flags.writeable)
            archive.payloads.clear();self.assertEqual(result.tobytes(),values.tobytes())
        archive=Archive({'v':np.arange(10,dtype='f4')});archive.payloads['v']=archive.payloads['v'][:-1]
        with self.assertRaisesRegex(ValueError,'truncated'):read_member(codec,archive,'v')

    def test_unknown_revision_falls_back_and_explicit_mode_rejects(self):
        codec=self.codec();auto=Reconstruction(codec,'auto')
        self.assertEqual(auto.mode,'reference')
        self.assertEqual(auto.decode(None,None,None,1,start=None,end=None,start_transformed=None,end_transformed=None),'reference')
        with self.assertRaisesRegex(ValueError,'not verified'):Reconstruction(codec,'numpy')


if __name__=='__main__':unittest.main()
