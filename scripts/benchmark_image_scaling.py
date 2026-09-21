#!/usr/bin/env python3
"""Measure resolution scaling and repeated resident-model imaging on Linux.

Timing, validation and packaging are separate. This runner never selects an
integration tolerance from the measured speed, extrapolates CPU times, or
reuses a rendered image as a computed repetition.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,os,re,signal,subprocess,threading,time
from pathlib import Path
import h5py
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def output(cmd):
    try:
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=20)
        return {'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:return {'command':cmd,'error':str(e)}


def gpu_apps():
    p=output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
    if p.get('returncode')!=0:raise RuntimeError('Cannot check GPU processes')
    return [r for r in csv.reader(p['stdout'].splitlines(),skipinitialspace=True) if r]


def physical_core_first(available,topology_root=Path('/sys/devices/system/cpu')):
    """Use one allowed hardware thread per physical core before SMT siblings."""
    groups={}
    for cpu in sorted(available):
        base=topology_root/f'cpu{cpu}'/'topology'
        try:
            core=(int((base/'physical_package_id').read_text()),
                  int((base/'core_id').read_text()))
        except (OSError,ValueError):
            core=('unknown',cpu)
        groups.setdefault(core,[]).append(cpu)
    return [cpus[i] for i in range(max(map(len,groups.values()),default=0))
            for cpus in groups.values() if i<len(cpus)]


def measure(cmd,env,path,gpu,allowed):
    if gpu:
        active=[r for r in gpu_apps() if r[1] not in allowed]
        if active:raise RuntimeError('Unrelated GPU computation present: '+repr(active))
    stop=threading.Event();sample_path=path.with_suffix('.memory.jsonl')
    process=None;peak_rss=0;peak_swap=0;peak_gpu=None;gpu_queries=[]
    samples=sample_path.open('w')
    def gpu_log(query,dest):
        f=dest.open('w')
        p=subprocess.Popen(['nvidia-smi',query,'--format=csv,noheader,nounits','--loop-ms=200'],stdout=f,stderr=subprocess.DEVNULL)
        gpu_queries.append((p,f))
    if gpu:
        gpu_log('--query-gpu=timestamp,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',path.with_suffix('.gpu.csv'))
        gpu_log('--query-compute-apps=timestamp,pid,used_gpu_memory',path.with_suffix('.gpu-process.csv'))
    with path.with_suffix('.log').open('w') as log:
        started=time.monotonic();process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        def watch():
            nonlocal peak_rss,peak_swap
            while not stop.is_set():
                try:
                    text=Path(f'/proc/{process.pid}/status').read_text()
                    values={k:int(v)*1024 for k,v in re.findall(r'^(VmRSS|VmHWM|VmSwap):\s+(\d+)\s+kB',text,re.M)}
                    values.update(monotonic_s=time.monotonic(),pid=process.pid)
                    peak_rss=max(peak_rss,values.get('VmHWM',0));peak_swap=max(peak_swap,values.get('VmSwap',0))
                    samples.write(json.dumps(values)+'\n');samples.flush()
                except FileNotFoundError:pass
                stop.wait(.05)
        thread=threading.Thread(target=watch);thread.start()
        try:
            _,status,usage=os.wait4(process.pid,0)
            elapsed=time.monotonic()-started
            process.returncode=os.waitstatus_to_exitcode(status)
        except BaseException:
            process.terminate();process.wait();raise
        finally:
            stop.set();thread.join();samples.close()
            for p,f in gpu_queries:p.terminate();p.wait();f.close()
    if gpu:
        for row in csv.reader(path.with_suffix('.gpu-process.csv').read_text().splitlines(),skipinitialspace=True):
            if len(row)>=3 and row[1].strip()==str(process.pid):
                try:peak_gpu=max(peak_gpu or 0,int(float(row[2])*2**20))
                except ValueError:pass
        if peak_gpu is None:raise RuntimeError('GPU memory samples missing for measured process')
    return dict(pid=process.pid,command=cmd,returncode=process.returncode,wall_s=elapsed,
                host_peak_rss_bytes=peak_rss,host_sampled_peak_rss_bytes=peak_rss,
                host_wait4_maxrss_bytes=int(usage.ru_maxrss*1024),
                host_sampled_peak_swap_bytes=peak_swap,gpu_sampled_peak_bytes=peak_gpu,
                memory_sampling_interval_s=.05,gpu_sampling_interval_s=.2,
                user_cpu_s=usage.ru_utime,system_cpu_s=usage.ru_stime)


def check_image(path,n,keep):
    with h5py.File(path) as h:
        frames=[name for name in h if name.startswith('frame_')]
        if frames != ['frame_0'] or [name for name in h['frame_0'] if name.startswith('freq_')] != ['freq_0']:
            raise ValueError('This benchmark requires one frame and one frequency per image')
        g=h['frame_0/freq_0'];im=np.stack([g[s][...] for s in 'IQUV'])
        reason=h['frame_0/diagnostics/reason'][...]
        steps=h['frame_0/diagnostics/total_steps'][...] if 'total_steps' in h['frame_0/diagnostics'] else None
        if im.shape!=(4,n,n) or not np.isfinite(im).all():raise ValueError('Invalid Stokes: '+str(path))
        if not np.all(reason==1):raise ValueError('Ray termination: '+str(path))
        peak=float(im[0].max());excess=float(np.maximum(np.sqrt(np.sum(im[1:]**2,axis=0))-im[0],0).max())
        if peak<=0 or im[0].min() < -peak*1e-12 or excess>peak*1e-10:raise ValueError('Unphysical Stokes')
        digest=hashlib.sha256(im.tobytes()).hexdigest()
        info=dict(native_sha256=sha(path),parameters_sha256=sha(str(path)+'.params'),stokes_sha256=digest,
                  minimum_intensity=float(im[0].min()),polarization_excess_over_peak=excess/peak,finite=True,
                  pixels=n*n,returned=int((reason==1).sum()))
        if steps is not None:info['steps']=dict(mean=float(steps.mean()),p99=float(np.percentile(steps,99)),maximum=int(steps.max()))
        if keep:
            slim=path.with_suffix('.stokes.npz');np.savez_compressed(slim,IQUV=im)
            info['stokes_archive']=str(slim);info['stokes_archive_sha256']=sha(slim)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bin-dir',type=Path,required=True);p.add_argument('--parameters',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',default='cuda');p.add_argument('--device-label',required=True)
    p.add_argument('--models',nargs='+',choices=['riaf','iharm','athenak'],default=['riaf','iharm','athenak'])
    p.add_argument('--resolutions',nargs='+',type=int,default=[256,512,1024,2048])
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--resident-count',type=int,default=16)
    p.add_argument('--allowed-gpu-process',action='append',default=[])
    p.add_argument('--cpu-set',help='CPU backends: comma-separated affinity; default fills physical cores before SMT siblings')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    if min(a.resolutions+[a.repeats])<1 or a.resident_count<0:p.error('positive sizes/repeats required')
    gpu=a.backend=='cuda';env=os.environ.copy();available=os.sched_getaffinity(0)
    affinity=physical_core_first(available)
    if a.cpu_set:
        affinity=[int(x) for x in a.cpu_set.split(',')]
        if len(set(affinity))!=len(affinity) or not set(affinity)<=available:
            p.error('CPU affinity must contain distinct allowed CPUs')
    if a.backend=='serial':affinity=affinity[:1]
    elif a.backend.startswith('omp'):
        n=int(a.backend[3:]);affinity=affinity[:n]
        if len(affinity)!=n:p.error('not enough CPUs')
        env.update(OMP_NUM_THREADS=str(n),OMP_PROC_BIND='spread',OMP_PLACES='threads')
    elif not gpu:p.error('backend must be cuda, serial or ompN')
    env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
    prefix=[] if gpu else ['taskset','-c',','.join(map(str,affinity))]
    report=dict(complete=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                device_label=a.device_label,backend=a.backend,driver_sha256=sha(__file__),rows=[],binaries={},parameters={},
                protocol='One warm-up at each resolution, then all measured repeats; actual executable time; resident cases recompute the same image with one loaded model and complete output, first image excluded from steady-state statistics.',
                gpu_memory_scope='Sampled per-process allocated memory, includes CUDA context; monitoring starts before launch',
                host_memory_scope='Maximum post-exec /proc/PID/status VmHWM observed; wait4 retained separately because it can include launcher memory; swap separately sampled',
                topology=output(['lscpu']),cpu_affinity=affinity,allowed_gpu_processes=a.allowed_gpu_process,
                gpu_before=output(['nvidia-smi','-q']) if gpu else None,
                environment={k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES']})
    def save():write_json(a.output/'benchmark.json',report)
    save()
    for model in a.models:
        binary=(a.bin_dir/('kpolaris_model_image_'+model)).resolve();par=(a.parameters/(model+'.par')).resolve()
        report['binaries'][model]=dict(path=str(binary),sha256=sha(binary),version=output([str(binary),'--version']))
        report['parameters'][model]=dict(path=str(par),sha256=sha(par))
        common=prefix+[str(binary),'--parameter_file='+str(par),'--parameter_output=auto']
        if model!='riaf':common += ['--'+('iharm_dump' if model=='iharm' else 'athenak_dump')+'='+str((a.data_root/('iharm.h5' if model=='iharm' else 'athenak.bin')).resolve())]
        hashes={}
        jobs=[(n,rep,1) for n in a.resolutions for rep in range(a.repeats+1)]
        if a.resident_count:jobs += [(256,rep,a.resident_count+1) for rep in range(1,a.repeats+1)]
        for n,rep,count in jobs:
            name=f'{model}_{n}_'+('resident' if count>1 else 'single')+f'_r{rep}'
            path=a.output/(name+'.h5');cmd=common+[f'--nx={n}',f'--ny={n}',f'--output={path}',f'--repeat_images={count}']
            print('START',name,flush=True);row=dict(name=name,model=model,nx=n,repeat=rep,warmup=rep==0,resident_count=count)
            row.update(measure(cmd,env,path,gpu,a.allowed_gpu_process));report['rows'].append(row);save()
            if row['returncode']:raise RuntimeError(name+' failed: see its log')
            log=path.with_suffix('.log').read_text();row['timing_s']={k:float(v) for k,v in re.findall(r'^timing\s+(\S+)\s+([-+0-9.eE]+)\s+s$',log,re.M)}
            targets=[path] if count==1 else [path.with_name(path.stem+f'_repeat{i:04d}'+path.suffix) for i in range(count)]
            row['images']=[]
            for i,target in enumerate(targets):
                info=check_image(target,n,keep=(rep==1 and i==0));row['images'].append(info)
                if n not in hashes:hashes[n]=info['stokes_sha256']
                if hashes[n]!=info['stokes_sha256']:raise ValueError('Repeated image changed at '+name)
            row['stokes_repeat_equal']=True
            if count>1:
                vals=[float(x) for x in re.findall(r'^timing repeat_image_elapsed (\S+) s$',log,re.M)]
                assert len(vals)==count
                row['resident_frame_s']=vals;row['steady_mean_s']=sum(vals[1:])/(count-1)
            save();print('DONE',name,round(row['wall_s'],4),'RSS_GiB',round(row['host_peak_rss_bytes']/2**30,3),'GPU_GiB',None if row['gpu_sampled_peak_bytes'] is None else round(row['gpu_sampled_peak_bytes']/2**30,3),flush=True)
    report['complete']=True;report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()

if __name__=='__main__':main()
