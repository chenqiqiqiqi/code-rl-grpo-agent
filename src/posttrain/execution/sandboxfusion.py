from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from posttrain.common import digest, file_digest
from posttrain.execution.contracts import CaseResult, Limits, Status


REQUIRED_PROBES = {
    'non_root','network_denied','host_canary_unreadable','secrets_absent',
    'answers_absent','process_limit','memory_limit','output_limit','timeout',
    'read_only_root','workspace_reset','cpu_limit','child_cleanup',
    'unreadable_workspace_cleanup','candidate_sigkill','batch_workspace_reset','batch_child_cleanup',
    'python_script_exit','stdin_regular_file',
}


def validate_attestation(path: Path, endpoint: str, limits: Limits) -> dict:
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('evidence_level') != 'isolated_execution' or record.get('endpoint') != endpoint:
        raise ValueError('attestation_endpoint_or_evidence_mismatch')
    now = datetime.now(timezone.utc)
    expiry = datetime.fromisoformat(record['expires_at'])
    if expiry.tzinfo is None or expiry <= now:
        raise ValueError('attestation_expired')
    if record.get('limits') != asdict(limits):
        raise ValueError('attestation_resource_limits_mismatch')
    probes = record.get('probes',{})
    if any(probes.get(k,{}).get('passed') is not True for k in REQUIRED_PROBES):
        raise ValueError('incomplete_isolation_probes')
    # Require real probe files, tied to immutable deployment identity.
    for name in REQUIRED_PROBES:
        evidence = path.parent / probes[name]['evidence_file']
        if not evidence.is_file() or file_digest(evidence) != probes[name]['sha256']:
            raise ValueError('probe_evidence_missing_or_modified')
    if not record.get('deployment_id') or not record.get('image_digest','').startswith('sha256:'):
        raise ValueError('missing_immutable_deployment_identity')
    return record


def parse_response(payload: dict, limits: Limits, *, supervised: bool = False) -> CaseResult:
    """Only known process outcomes count. Unknown/SandboxError are infrastructure."""
    infra = lambda reason: CaseResult(Status.INFRA,physical_executions=None,detail=reason)
    if not isinstance(payload,dict) or payload.get('status') not in ('Success','Failed'):
        return infra('unknown_or_sandbox_error')
    compiled = payload.get('compile_result')
    result = payload.get('run_result')
    if compiled and (compiled.get('status') != 'Finished' or compiled.get('return_code') != 0):
        # Python has no separate compile step; an unexpected compile failure is untrusted.
        return infra('unexpected_compile_result')
    if not isinstance(result,dict) or result.get('status') not in ('Finished','TimeLimitExceeded'):
        return infra('missing_or_unknown_process_result')
    stdout,stderr = result.get('stdout',''),result.get('stderr','')
    if not isinstance(stdout,str) or not isinstance(stderr,str):
        return infra('malformed_output')
    elapsed = result.get('execution_time')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed < 0:
        return infra('missing_process_timing')
    retcode = result.get('return_code')
    event=payload.get('project_event') if supervised else None
    successful = result['status']=='Finished' and type(retcode) is int and retcode==0 and event is None
    if (payload['status']=='Success') != successful:
        return infra('inconsistent_success_status')
    if event=='output_limit':
        status=Status.OUTPUT
    elif result['status']=='TimeLimitExceeded' or event=='program_timeout':
        status=Status.TIMEOUT
    elif type(retcode) is not int:
        return infra('missing_exit_code')
    elif retcode==0:
        status=Status.OK
    elif retcode in (-9,137):
        # Only a healthy trusted local supervisor can attribute this to its child.
        # A candidate must not invalidate a whole GRPO group by killing itself.
        if not supervised:
            return infra('unattributed_sigkill')
        status=Status.RUNTIME
    elif 'SyntaxError' in stderr or 'IndentationError' in stderr:
        status=Status.SYNTAX
    elif 'MemoryError' in stderr:
        status=Status.MEMORY
    else:
        status=Status.RUNTIME
    truncated = event=='output_limit' or len(stdout.encode('utf-8'))+len(stderr.encode('utf-8'))>limits.output_bytes
    if truncated:
        status=Status.OUTPUT
        stdout=stdout.encode('utf-8')[:limits.output_bytes].decode('utf-8',errors='replace')
        stderr=stderr.encode('utf-8')[:limits.output_bytes].decode('utf-8',errors='replace')
    # Stock SandboxFusion has no queue or physical execution telemetry. Keep unknown.
    return CaseResult(status,stdout,stderr,retcode,float(elapsed),None,None,truncated)


class SandboxFusionBackend:
    evidence_level = 'isolated_execution'

    def __init__(self, endpoint: str, attestation: Path, limits: Limits):
        parsed=urlsplit(endpoint)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('endpoint_must_not_embed_credentials')
        if parsed.scheme not in ('http','https') or not parsed.hostname:
            raise ValueError('invalid_endpoint')
        if parsed.scheme=='http' and parsed.hostname not in ('localhost','127.0.0.1','::1'):
            raise ValueError('remote_endpoint_requires_https_or_ssh_tunnel')
        self.endpoint=endpoint.rstrip('/')
        self.attestation_path=attestation
        self.record=validate_attestation(attestation,self.endpoint,limits)
        self.version='sandboxfusion:'+self.record['deployment_id']+':'+digest(self.record)
        self.limits=limits

    def run(self, code: str, stdin: str, limits: Limits, request_id: str, timeout_s: float) -> CaseResult:
        # Recheck expiry on every execution; attestation is never a permanent bypass.
        record=validate_attestation(self.attestation_path,self.endpoint,self.limits)
        if record != self.record or replace(limits,time_s=self.limits.time_s) != self.limits or limits.time_s>self.limits.time_s:
            raise ValueError('deployment_or_resource_config_changed_revalidate')
        body={'language':'python','code':code,'stdin':stdin,'run_timeout':limits.time_s,
              'memory_limit_MB':limits.memory_mb,'files':{},'fetch_files':[]}
        request=urllib.request.Request(self.endpoint+'/run_code',data=json.dumps(body).encode(),
                                       headers={'Content-Type':'application/json','X-Request-ID':request_id})
        try:
            with urllib.request.urlopen(request,timeout=min(timeout_s,limits.time_s+30)) as response:
                # A project gateway must pin deployment identity and bound output server-side.
                if response.headers.get('X-Deployment-ID') != self.record['deployment_id']:
                    return CaseResult(Status.INFRA,physical_executions=None,detail='deployment_identity_mismatch')
                raw=response.read(limits.output_bytes*4+65537)
                if len(raw)>limits.output_bytes*4+65536:
                    return CaseResult(Status.INFRA,physical_executions=None,detail='unbounded_service_response')
                return parse_response(json.loads(raw),limits)
        except (urllib.error.URLError,TimeoutError,OSError,ValueError) as exc:
            # A network timeout may leave work running; duplicate cost remains unknown.
            return CaseResult(Status.INFRA,physical_executions=None,detail=type(exc).__name__)
