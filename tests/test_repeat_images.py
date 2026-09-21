#!/usr/bin/env python3
"""Check resident fast-light repetitions against ordinary image execution."""
import argparse,subprocess,tempfile
from pathlib import Path
import h5py
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--image-exe',type=Path,required=True)
p.add_argument('--parameters',type=Path,required=True)
p.add_argument('--workdir',type=Path,required=True)
a=p.parse_args();a.workdir.mkdir(parents=True,exist_ok=True)
workspace=tempfile.TemporaryDirectory(prefix="repeat-",dir=a.workdir)
a.workdir=Path(workspace.name)
common=[str(a.image_exe.resolve()),'--parameter_file='+str(a.parameters.resolve()),'--nx=16','--ny=16','--timing=1','--parameter_output=auto']
def run(name,extra,ok=True):
 cmd=common+['--output='+str((a.workdir/name).resolve())]+extra
 r=subprocess.run(cmd,text=True,capture_output=True)
 (a.workdir/(name+'.log')).write_text(r.stdout+r.stderr)
 assert (r.returncode==0)==ok,(cmd,r.stdout[-500:],r.stderr)
 return r
run('ordinary.h5',[])
r=run('sequence.h5',['--repeat_images=3'])
assert r.stdout.count('repeat_image_begin ')==3
assert r.stdout.count('timing image_kernel_single ')==3
assert r.stdout.count('timing repeat_image_elapsed ')==3
with h5py.File(a.workdir/'ordinary.h5') as reference:
 for i in range(3):
  path=a.workdir/f'sequence_repeat{i:04d}.h5'
  assert Path(str(path)+'.params').is_file()
  with h5py.File(path) as image:
   for key in ['frame_0/freq_0/'+s for s in 'IQUV']+['frame_0/diagnostics/reason']:
    assert np.array_equal(image[key][...],reference[key][...]),(i,key)
run('sequence.h5',['--repeat_images=3'],False)
run('invalid.h5',['--repeat_images=0'],False)
run('slow.h5',['--repeat_images=2','--slow_light=1'],False)
run('params.h5',['--repeat_images=2','--parameter_output=shared.par'],False)
print('Repeated images equal ordinary image; all three kernels run; output collisions and unsupported modes rejected.')
