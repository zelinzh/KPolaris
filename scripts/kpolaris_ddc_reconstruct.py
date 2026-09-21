"""Allocation-bounded reconstruction for the verified external DDC formula.

No archive format or arithmetic change: CRC/preconditioning stay in the codec
reader, float32 operations retain their order (no fused multiply-add), and all
published outputs own their storage. Unknown codec revisions use the reference.
"""
from __future__ import annotations
import hashlib
import inspect
import io
import math
import threading

# Narrow compatibility gate: reconstruction optimizations must be revalidated
# when the external formula/reader changes. Auto falls back rather than guessing.
_VERIFIED = {
    'transformed':'51f978054c5feae224f047119dccb42dabc7fa8701e72f9bb6c725a75cf2e796',
    'unpack_int4':'be01f592aac3df5acd13a01ba1516881d7482010cf08e5062a8d629e1d03f6fb',
    'decode_dataset':'dcf84029358ad2450b900a8c513b8e2e03597513e934ff80a084a874cd16f0dd',
    'np_load_member':'3f8d1fab84a996c24712c9b84da7c18c6b83536ea780102a44e7c93521930f1d',
    'read_quantized':'e2c8efa216cf3187554b6746e230976c80892e501d619a3743e80f397fa4c4a8',
    'dequantize_residual':'a9ea700b3cd055e5d6ef5b657504c4c95a2a01f3766d50f04731e00bca121916',
    'inverse_transformed':'373cbd5941545d644105e44c91fb523082d8076de3407d4c029bdeaab46011a1',
}


def verified_codec(codec):
    try:
        for name,digest in _VERIFIED.items():
            if hashlib.sha256(inspect.getsource(getattr(codec,name)).encode()).hexdigest()!=digest:
                return False
        return codec.inverse_transformed.__globals__['LOG_SPACE_DATASETS']==frozenset(('prims.rho','prims.u'))
    except (AttributeError,KeyError,OSError,TypeError):
        return False


def read_member(codec, archive, name):
    """Read-only numeric NPY view retaining the validated archive payload owner."""
    import numpy as np
    payload=archive.read(name)  # Includes the codec's size and CRC checks.
    header=io.BytesIO(payload)
    version=np.lib.format.read_magic(header)
    if version==(1,0):
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(header)
    elif version==(2,0):
        shape,fortran,dtype=np.lib.format.read_array_header_2_0(header)
    else:
        return codec.np_load_member(archive,name)
    if dtype.hasobject:
        raise ValueError('Object arrays cannot be loaded when allow_pickle=False')
    count=math.prod(shape)
    offset=header.tell()
    if count<0 or dtype.itemsize==0 or count*dtype.itemsize>len(payload)-offset:
        raise ValueError(f'Invalid or truncated numeric NPY member: {name}')
    return np.frombuffer(payload,dtype=dtype,count=count,offset=offset).reshape(shape,order='F' if fortran else 'C')


class Reconstruction:
    def __init__(self, codec, mode='auto'):
        if mode not in ('auto','reference','numpy'):
            raise ValueError('reconstruction must be auto, reference or numpy')
        self.codec=codec
        compatible=verified_codec(codec) if mode!='reference' else False
        if mode=='numpy' and not compatible:
            raise ValueError('External DDC reconstruction revision is not verified; use auto or reference')
        self.mode='numpy' if compatible else 'reference'
        self._local=threading.local()

    def decode(self, archive, metadata, dataset, frame_index, *, start, end,
               start_transformed, end_transformed):
        if self.mode=='reference':
            return self.codec.decode_dataset(archive,metadata,dataset,frame_index,
                start=start,end=end,start_transformed=start_transformed,end_transformed=end_transformed)
        import numpy as np
        codec=self.codec
        # Use the reference for exotic inputs rather than changing NumPy promotion.
        if start_transformed.dtype!=np.dtype('float32') or end_transformed.dtype!=np.dtype('float32') or start_transformed.shape!=end_transformed.shape:
            return codec.decode_dataset(archive,metadata,dataset,frame_index,
                start=start,end=end,start_transformed=start_transformed,end_transformed=end_transformed)
        shape=start_transformed.shape;size=start_transformed.size
        buffer=getattr(self._local,'buffer',None)
        if buffer is None or buffer.size<size:
            buffer=np.empty(size,dtype=np.float32);self._local.buffer=buffer
        scratch=buffer[:size].reshape(shape)
        # Output storage is never the reusable scratch: service/cache consumers
        # can retain it after the worker starts decoding the next frame.
        result=np.empty(shape,dtype=np.float32)
        tau=codec.frame_tau(archive,metadata,(dataset,),frame_index)
        np.multiply(start_transformed,1.0-tau,out=result)
        np.multiply(end_transformed,tau,out=scratch)
        np.add(result,scratch,out=result)
        bits=codec.dataset_bits(metadata,dataset)
        member=lambda suffix:codec.archive_member_name(dataset,frame_index,suffix)
        if bits==4:
            packed=read_member(codec,archive,member('q4_packed'))
            qshape=tuple(int(v) for v in read_member(codec,archive,member('q4_shape')))
            quantized=codec.unpack_int4(packed,qshape)
        else:
            quantized=read_member(codec,archive,member(f'q{bits}'))
        scale=read_member(codec,archive,member('scale'))
        if quantized.shape!=shape:
            raise ValueError('DDC quantized shape disagrees with anchor shape')
        if scale.dtype==np.dtype('float32') and quantized.dtype in (np.dtype('int8'),np.dtype('int16')):
            # int8/int16 -> float32 is exact; match cast then multiply while
            # avoiding a full-sized cast array. No operation is reassociated.
            np.multiply(quantized,scale,out=scratch,dtype=np.float32)
        else:
            scratch[...] = (quantized.astype(np.float32)*scale).astype(np.float32,copy=False)
        indices_name=member('exception_indices')
        if indices_name in archive.namelist():
            indices=read_member(codec,archive,indices_name)
            values=read_member(codec,archive,member('exception_values'))
            scratch.reshape(-1)[indices.astype(np.int64)] = values
        np.add(result,scratch,out=result)
        if dataset in ('prims.rho','prims.u'):
            np.exp(result,out=result,dtype=np.float32)
        return result
