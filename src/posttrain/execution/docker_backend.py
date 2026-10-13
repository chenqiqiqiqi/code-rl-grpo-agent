from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict,replace
from pathlib import Path

from posttrain.execution.contracts import CaseResult, Limits, Status
from posttrain.execution.sandboxfusion import parse_response,validate_attestation


def container_command(image: str, name: str, limits: Limits) -> list[str]:
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',image):
        raise ValueError('use_locally_inspected_immutable_image_id')
    scope=os.environ.get('POSTTRAIN_EXECUTION_RUN')
    if scope is not None and not re.fullmatch(r'[0-9a-f]{32}',scope):
        raise ValueError('invalid_execution_run_scope')
    labels=['--label','posttrain.execution_run='+scope] if scope is not None else []
    return ['docker','run',*labels,'--name',name,'--interactive','--network=none','--read-only',
            '--cap-drop=ALL','--cap-add=SETUID','--cap-add=SETGID','--cap-add=KILL',
            '--cap-add=DAC_OVERRIDE','--cap-add=FOWNER','--security-opt=no-new-privileges:true',
            '--pids-limit',str(limits.pids+4),'--cpus',str(limits.cpu_cores),
            '--memory',f'{limits.memory_mb+256}m','--memory-swap',f'{limits.memory_mb+256}m',
            '--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=33554432,mode=1777',
            '--ulimit','nofile=64:64','--log-driver=none',image]


class DockerBackend:
    evidence_level='isolated_execution'

    def __init__(self,image: str,limits: Limits,attestation: Path):
        if sys.platform!='linux':
            raise RuntimeError('Linux execution host required; Windows is development only')
        self.image,self.limits=image,limits
        self.attestation=attestation
        self.record=validate_attestation(attestation,'docker://'+image,limits)
        self.version='sandboxfusion-docker:'+image+':'+self.record['deployment_id']

    def run(self,code,stdin,limits,request_id,timeout_s):
        return self.run_batch(code,[stdin],limits,request_id,timeout_s)[0]

    def run_batch(self,code,stdins,limits,request_id,timeout_s):
        validate_attestation(self.attestation,'docker://'+self.image,self.limits)
        if replace(limits,time_s=self.limits.time_s)!=self.limits or limits.time_s>self.limits.time_s:
            raise ValueError('limits_must_match_attested_configuration')
        return execute_batch(self.image,code,stdins,limits,timeout_s)[0]


def execute_container(image: str, code: str, stdin: str, limits: Limits, timeout_s: float) -> tuple[CaseResult,dict]:
    results,evidence=execute_batch(image,code,[stdin],limits,timeout_s)
    evidence['result']=asdict(results[0])
    return results[0],evidence


def execute_batch(image: str, code: str, stdins: list[str], limits: Limits, timeout_s: float) -> tuple[list[CaseResult],dict]:
    """Internal primitive used only after attestation, or for fixed built-in probes.

    Each container has no mounts, no inherited env, no network, no device passthrough.
    Even if controller cleanup fails, the stage is invalid and cannot be rewarded.
    """
    if sys.platform!='linux':
        raise RuntimeError('Linux execution host required')
    name='posttrain-'+uuid.uuid4().hex
    cmd=container_command(image,name,limits)
    if not stdins or len(stdins)>4096 or not all(isinstance(x,str) for x in stdins):
        raise ValueError('invalid_batch_inputs')
    payload={'code':code,'stdins':stdins,'limits':asdict(limits)}
    started=time.monotonic()
    evidence={'container':name,'image':image,'command':cmd}
    def failed(reason):
        return [CaseResult(Status.INFRA,physical_executions=None,detail=reason) for _ in stdins]
    results=failed('container_not_started')
    try:
        # The trusted worker bounds candidate output. Controller output is a small JSON record.
        proc=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=timeout_s)
        evidence['return_code']=proc.returncode
        inspect=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
        info=json.loads(inspect.stdout)[0] if inspect.returncode==0 else None
        evidence['inspect']=info
        if info and info['State'].get('OOMKilled'):
            # Whole worker OOM is infrastructure: no candidate-specific attribution.
            results=failed('worker_container_oom')
        elif proc.returncode==0:
            response=json.loads(proc.stdout)
            evidence['response']=response
            if response.get('worker_complete') is not True or len(response.get('cases',[]))!=len(stdins):
                results=failed('incomplete_worker_batch')
            else:
                results=[replace(parse_response(row,limits,supervised=True),queue_s=0.0,physical_executions=1)
                         for row in response['cases']]
        else:
            evidence['controller_stderr']=proc.stderr[-4000:]
            results=failed('worker_exit')
    except (OSError,subprocess.TimeoutExpired,ValueError,KeyError) as exc:
        results=failed(type(exc).__name__)
    finally:
        try:
            cleanup=subprocess.run(['docker','rm','--force',name],capture_output=True,text=True,timeout=15)
            evidence['cleanup_return_code']=cleanup.returncode
            if cleanup.returncode!=0:
                results=failed('container_cleanup_failed')
        except (OSError,subprocess.TimeoutExpired):
            results=failed('container_cleanup_unknown')
    evidence['wall_time_s']=time.monotonic()-started
    evidence['results']=[asdict(result) for result in results]
    return results,evidence
