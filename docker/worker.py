"""Trusted same-candidate batch worker, ONLY in the disposable Linux image.

Uses pinned SandboxFusion run_commands and response models, with a project adapter
for bounded asynchronous output, hard resource limits, child UID and process cleanup.
Expected answers are never part of this process's input.
"""
import asyncio
import json
import math
import os
import resource
import shlex
import signal
import sys
import tempfile
import time
import types
from pathlib import Path
import shutil


async def main():
    if sys.platform != 'linux' or os.getuid()!=0 or not os.path.exists('/.dockerenv'):
        raise RuntimeError('worker_requires_isolated_linux_container')
    payload=json.loads(sys.stdin.buffer.readline(64*1024*1024))
    if set(payload) != {'code','stdins','limits'} or not 0<len(payload['stdins'])<=4096:
        raise ValueError('unexpected_payload_fields')
    limits=payload['limits']
    output_limit=limits['output_bytes']
    # Redirect library logging away from the machine-readable response channel.
    original_stdout=sys.stdout
    sys.stdout=sys.stderr
    # Load only SandboxFusion's audited command runner; optional language runners
    # and dataset plugins would add unrelated imports and enlarge the trusted base.
    sys.path.insert(0,'/opt/SandboxFusion')
    runners=types.ModuleType('sandbox.runners')
    runners.__path__=['/opt/SandboxFusion/sandbox/runners']
    sys.modules['sandbox.runners']=runners
    import structlog
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    from sandbox.runners import base
    from sandbox.runners.types import CodeRunArgs, CommandRunResult, CommandRunStatus
    event=None

    async def terminate_candidates():
        # A child may setsid() and leave the original process group. UID scanning
        # stays inside this container's PID namespace and also covers that case.
        import psutil
        deadline=time.monotonic()+3
        while True:
            alive=[]
            for process in psutil.process_iter(['pid','uids','status']):
                try:
                    if process.info['uids'].real==10001 and process.info['status']!=psutil.STATUS_ZOMBIE:
                        alive.append(process)
                        process.kill()
                except psutil.NoSuchProcess:
                    pass
            if not alive:
                return
            if time.monotonic()>deadline:
                raise RuntimeError('candidate_cleanup_failed')
            await asyncio.sleep(.01)

    async def bounded_command(command, timeout=10, stdin=None, cwd=None, extra_env=None,
                              use_exec=False, preexec_fn=None):
        nonlocal event
        def prepare_child():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            mem=limits['memory_mb']*1024*1024
            resource.setrlimit(resource.RLIMIT_AS,(mem,mem))
            resource.setrlimit(resource.RLIMIT_FSIZE,(output_limit,output_limit))
            resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
            resource.setrlimit(resource.RLIMIT_NPROC,(limits['pids'],limits['pids']))
            cpu=max(1,math.ceil(timeout))
            resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu+1))
            os.setgroups([])
            os.setgid(10001)
            os.setuid(10001)
        args=command if isinstance(command,list) else shlex.split(command)
        started=time.monotonic()
        # Contest programs sometimes use fstat(0).st_size. Provide an unlinked,
        # read-only regular file for this case only, not a pipe or future inputs.
        with tempfile.NamedTemporaryFile(dir='/tmp') as input_data:
            input_data.write((stdin or '').encode());input_data.flush()
            with open(input_data.name,'rb') as input_stream:
                input_data.close()
                proc=await asyncio.create_subprocess_exec(*args,cwd=cwd,stdin=input_stream,
                    stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,preexec_fn=prepare_child,
                    env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','LANG':'C.UTF-8'})
        outputs=[bytearray(),bytearray()]
        total=0
        def terminate():
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        async def read(stream,index):
            nonlocal total,event
            while chunk:=await stream.read(4096):
                room=max(0,output_limit-total)
                outputs[index].extend(chunk[:room])
                total+=len(chunk)
                if total>output_limit:
                    event='output_limit'
                    terminate()
        tasks=[asyncio.create_task(read(proc.stdout,0)),asyncio.create_task(read(proc.stderr,1))]
        status=CommandRunStatus.Finished
        try:
            await asyncio.wait_for(proc.wait(),timeout)
        except asyncio.TimeoutError:
            event='program_timeout'
            status=CommandRunStatus.TimeLimitExceeded
        finally:
            terminate()
            await terminate_candidates()
            await proc.wait()
            try: await asyncio.wait_for(asyncio.gather(*tasks),2)
            except asyncio.TimeoutError:
                for task in tasks: task.cancel()
            # PID 1 adopts orphaned descendants; do not carry zombies into the
            # next case or gradually consume its process allowance.
            try:
                while os.waitpid(-1,os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
        if event is None and proc.returncode==-signal.SIGXCPU:
            event='program_timeout';status=CommandRunStatus.TimeLimitExceeded
        return CommandRunResult(status=status,execution_time=time.monotonic()-started,
            return_code=proc.returncode,stdout=outputs[0].decode(errors='replace'),stderr=outputs[1].decode(errors='replace'))

    # Version-specific override; upstream source/hash is pinned in configs/sources.json.
    base.run_command_bare=bounded_command
    responses=[]
    for stdin in payload['stdins']:
        event=None
        with tempfile.TemporaryDirectory(prefix='candidate-',dir='/tmp') as tmp:
            os.chmod(tmp,0o777)
            source=os.path.join(tmp,'solution.py')
            with open(source,'w',encoding='utf-8') as f: f.write(payload['code'])
            os.chmod(source,0o444)
            args=CodeRunArgs(code=payload['code'],stdin=stdin,run_timeout=limits['time_s'],memory_limit_MB=limits['memory_mb'])
            # -I keeps isolated environment/user-site behavior; normal trusted
            # image site initialization preserves Python's exit()/quit().
            result=await base.run_commands(None,shlex.join([sys.executable,'-I',source]),tmp,{},args)
            run=result.run_result
            status=('SandboxError' if run.status==CommandRunStatus.Error else
                    'Success' if run.status==CommandRunStatus.Finished and run.return_code==0 and event is None else 'Failed')
            responses.append({'status':status,'message':'',**result.model_dump(),
                              'project_event':event,'physical_executions':1,'queue_s':0.0})
        # Clear the entire writable filesystem between tests, including paths a
        # candidate created outside its cwd. There are no mounted host files.
        for path in Path('/tmp').iterdir():
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)
    # Only this trusted UID can write the final response; child uses a distinct UID.
    original_stdout.write(json.dumps({'worker_complete':True,'cases':responses})+'\n')
    original_stdout.flush()


if __name__=='__main__':
    asyncio.run(main())
