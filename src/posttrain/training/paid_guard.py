"""Approval validation and an independent AutoDL shutdown deadline.

No resource creation/payment. The approval file is absent until the user approves
the concrete quote; billing start must come from the provider's actual start time.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from posttrain.common import file_digest,utc_now,write_json


def approved_deadline(row,resource_id,billing_start,now):
    if (row.get('authorized') is not True or not row.get('user_confirmation')
        or row.get('provider')!='autodl' or not resource_id or row.get('resource_id')!=resource_id
        or row.get('gpu_count')!=1 or row.get('auto_shutdown_authorized') is not True):
        raise ValueError('explicit_matching_single_GPU_authorization_required')
    if (not 0<row.get('hourly_cny',0)<=8 or not 0<row.get('max_gpu_hours',0)<=8
        or row['hourly_cny']*row['max_gpu_hours']>row.get('gpu_spend_cap_cny',0)):
        raise ValueError('approved_GPU_price_or_duration_cap_exceeded')
    if billing_start.tzinfo is None or now.tzinfo is None or billing_start>now:
        raise ValueError('actual_provider_billing_start_required')
    deadline=billing_start+timedelta(hours=row['max_gpu_hours'])
    if now>=deadline:raise ValueError('authorization_window_expired_shutdown_manually')
    return deadline


def best_effort_audit(path,row):
    try:write_json(path,row)
    except OSError:
        # A full or failed disk must never prevent the attempt to stop billing.
        try:print('Shutdown audit could not be persisted.',file=sys.stderr,flush=True)
        except OSError:pass


def stop(output,reason):
    best_effort_audit(output/('shutdown-'+reason+'.json'),{'requested_at':utc_now(),'reason':reason,
        'provider_shutdown_verified':False})
    # AutoDL documents this command inside its managed container. Caller verifies
    # AutoDLContainerUUID before reaching here; never run against a local VM.
    result=subprocess.run(['/usr/bin/shutdown'],capture_output=True,text=True,timeout=20)
    best_effort_audit(output/('shutdown-'+reason+'-return.json'),{'return_code':result.returncode,
        'returned_at':utc_now(),'provider_shutdown_verified':False,'stdout':result.stdout[-1000:],
        'stderr':result.stderr[-1000:]})
    # Exit code alone is NOT billing proof. Console/API readback is still required.
    return result.returncode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--billing-start',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watchdog',action='store_true');parser.add_argument('--watchdog-id')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if sys.platform!='linux' or not os.environ.get('AutoDLContainerUUID'):
        raise RuntimeError('managed_AutoDL_container_only')
    approval=json.loads(args.approval.read_text(encoding='utf-8'))
    start=datetime.fromisoformat(args.billing_start)
    now=datetime.now(timezone.utc)
    deadline=approved_deadline(approval,os.environ['AutoDLContainerUUID'],start,now)
    if args.watchdog:
        if not args.watchdog_id or not args.output.is_dir():raise ValueError('parent_watchdog_identity_required')
        # Monotonic bound also protects against a backward wall-clock adjustment.
        monotonic_end=time.monotonic()+(deadline-now).total_seconds()
        write_json(args.output/'watchdog-ready.json',{'pid':os.getpid(),'deadline':deadline.isoformat(),
            'watchdog_id':args.watchdog_id})
        while time.monotonic()<monotonic_end and datetime.now(timezone.utc)<deadline:
            time.sleep(min(5,max(.1,monotonic_end-time.monotonic())))
        raise SystemExit(stop(args.output,'hard_deadline'))
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:raise ValueError('job_command_required')
    # Exclusive creation rejects all stale readiness files before spawning work.
    try:args.output.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        # Do not interrupt a possibly active invocation owning this directory.
        raise
    except OSError:
        stop(args.output,'output_directory_failed')
        raise
    try:
        raise SystemExit(run_approved_job(args,approval,deadline,command))
    finally:
        stop(args.output,'job_finished_or_failed')


def run_approved_job(args,approval,deadline,command):
    watchdog_id=uuid.uuid4().hex
    watchdog_cmd=[sys.executable,'-m','posttrain.training.paid_guard','--approval',str(args.approval.resolve()),
        '--billing-start',args.billing_start,'--output',str(args.output.resolve()),'--watchdog','--watchdog-id',watchdog_id]
    with (args.output/'watchdog.log').open('ab') as log:
        watchdog=subprocess.Popen(watchdog_cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    ready=args.output/'watchdog-ready.json'
    wait_end=time.monotonic()+10
    while not ready.exists() and time.monotonic()<wait_end and watchdog.poll() is None:time.sleep(.1)
    try:readiness=json.loads(ready.read_text(encoding='utf-8'))
    except (OSError,ValueError):readiness={}
    if (watchdog.poll() is not None or readiness!={'pid':watchdog.pid,'deadline':deadline.isoformat(),
                                                'watchdog_id':watchdog_id}):
        raise RuntimeError('independent_shutdown_watchdog_failed')
    write_json(args.output/'job.json',{'started_at':utc_now(),'provider_billing_start':args.billing_start,
        'deadline':deadline.isoformat(),'command':command,'approval_sha256':file_digest(args.approval),
        'resource_id':approval['resource_id'],'hourly_cny':approval['hourly_cny'],'watchdog_pid':watchdog.pid})
    proc=None
    try:
        with (args.output/'job.log').open('ab') as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
            graceful_at=deadline-timedelta(minutes=10)
            while proc.poll() is None and datetime.now(timezone.utc)<graceful_at:time.sleep(1)
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=max(1,(deadline-datetime.now(timezone.utc)).total_seconds()-30))
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        write_json(args.output/'job-result.json',{'finished_at':utc_now(),'return_code':proc.returncode,
            'provider_billing_stopped_verified':False})
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
    return proc.returncode


if __name__=='__main__':main()
