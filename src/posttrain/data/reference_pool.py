"""Disjoint reference shards, bounded Linux workers, and fail-closed collection."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from posttrain.common import canonical,digest,file_digest,read_jsonl,utc_now,write_json,write_jsonl
from posttrain.data.schema import Problem


def prepare(source,output,*,workers=4,max_references=3):
    if type(workers) is not int or not 1<=workers<=4 or type(max_references) is not int or max_references<1:
        raise ValueError('invalid_reference_pool_limits')
    # Keep only metadata in memory; the full pool contains many unused source
    # references and must not require multiple GB of Python objects to partition.
    original_hash=file_digest(source)
    metadata=[]
    for row in read_jsonl(source):
        p=Problem.from_dict(row)
        metadata.append((p.problem_id,len(p.tests)*min(max_references,len(p.references))))
    if len(metadata)<workers or len({p[0] for p in metadata})!=len(metadata):
        raise ValueError('need_unique_problems_at_least_worker_count')
    output.mkdir(parents=True,exist_ok=False)
    partitions=[[] for _ in range(workers)];loads=[0]*workers
    # Balance only frozen test/reference counts, never observed success rates.
    for index in sorted(range(len(metadata)),key=lambda i:(-metadata[i][1],i)):
        selected=min(range(workers),key=lambda i:(loads[i],i))
        partitions[selected].append(index)
        loads[selected]+=metadata[index][1]
    assignments={index:worker for worker,indices in enumerate(partitions) for index in indices}
    inputs=output/'inputs';inputs.mkdir()
    with ExitStack() as stack:
        files=[stack.enter_context((inputs/f'shard-{i:02d}.jsonl').open('w',encoding='utf-8',newline='\n')) for i in range(workers)]
        for index,row in enumerate(read_jsonl(source)):
            files[assignments[index]].write(canonical(Problem.from_dict(row).to_dict())+'\n')
    if file_digest(source)!=original_hash:raise ValueError('source_changed_during_partition')
    shards=[]
    for index,indices in enumerate(partitions):
        name=f'shard-{index:02d}';path=output/'inputs'/(name+'.jsonl')
        shards.append({'input':path.relative_to(output).as_posix(),'sha256':file_digest(path),
            'run':'jobs/'+name,'problems':len(indices),'max_planned_case_judgments':loads[index]})
    manifest={'created_at':utc_now(),'source_sha256':original_hash,'input_problems':len(metadata),
        'workers':workers,'max_references':max_references,'shards':shards,
        'evidence_level':'static_partition_only','reference_executions':0}
    write_json(output/'pool.json',manifest)
    return manifest


def local_path(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('pool_path_outside_output')
    return path


def checked_manifest(output):
    manifest=json.loads((output/'pool.json').read_text(encoding='utf-8'))
    ids=[]
    if len(manifest['shards'])!=manifest['workers']:raise ValueError('pool_worker_count_changed')
    for shard in manifest['shards']:
        path=local_path(output,shard['input']);local_path(output,shard['run'])
        if file_digest(path)!=shard['sha256']:raise ValueError('shard_input_changed')
        shard_ids=[r['problem_id'] for r in read_jsonl(path)]
        if len(shard_ids)!=shard['problems']:raise ValueError('shard_problem_count_changed')
        ids.extend(shard_ids)
    if len(ids)!=manifest['input_problems'] or len(ids)!=len(set(ids)):
        raise ValueError('pool_coverage_or_overlap_error')
    return manifest


def collect(output):
    from posttrain.execution.service import checked_reply
    manifest=checked_manifest(output);accepted=[];statuses=[];receipts=[];identities=[]
    for shard in manifest['shards']:
        run=local_path(output,shard['run']);status_path=run/'status.json'
        if not status_path.exists():raise ValueError('incomplete_shard:'+shard['run'])
        status=json.loads(status_path.read_text(encoding='utf-8'))
        if not status.get('complete'):raise ValueError('incomplete_shard:'+shard['run'])
        identity=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))['identity']
        if identity['source_sha256']!=shard['sha256'] or identity['max_references']!=manifest['max_references']:
            raise ValueError('shard_run_identity_mismatch')
        identities.append((identity['executor_version'],identity['scoring_code_sha256']))
        problems={p.problem_id:p for p in (Problem.from_dict(r) for r in read_jsonl(local_path(output,shard['input'])))}
        record_files=sorted((run/'records').glob('*.json'))
        attempts=[json.loads(p.read_text(encoding='utf-8')) for p in record_files]
        physical=[row['result']['physical_executions'] for attempt in attempts for row in attempt['audit'] if 'result' in row]
        unknown=sum(not a.get('physical_accounting_complete',a['complete']) for a in attempts)
        known=sum(n for n in physical if n is not None)
        accounted=known if not unknown and all(n is not None for n in physical) else None
        if (status['physical_executions']!=accounted or status.get('known_physical_executions',known)!=known
            or status.get('unknown_physical_attempts',unknown)!=unknown):
            raise ValueError('shard_physical_accounting_mismatch')
        records=[r for r in attempts if r['complete']]
        if len(records)!=len(problems) or {r['problem_id'] for r in records}!=problems.keys():
            raise ValueError('shard_result_coverage_error')
        shard_accepted=[]
        for record in records:
            problem=problems[record['problem_id']];passing_codes=set()
            results=[r for r in record['audit'] if 'result' in r]
            if not results:raise ValueError('missing_reference_judgments')
            for row in results:
                index=row['reference_index']
                if type(index) is not int or not 0<=index<min(len(problem.references),manifest['max_references']):
                    raise ValueError('unregistered_reference_index')
                code=problem.references[index];raw=row['result']
                result=checked_reply(raw,raw['request_id'],code,problem.tests_hash,identity['executor_version'],len(problem.tests))
                if not result.valid:raise ValueError('completed_problem_has_invalid_judgment')
                if result.passed==result.total:passing_codes.add(code)
            if bool(record['accepted'])!=bool(passing_codes):raise ValueError('reference_acceptance_mismatch')
            if record['accepted']:
                p=Problem.from_dict(record['accepted'])
                if (len(p.references)!=1 or p.references[0] not in passing_codes or
                    digest(p.to_dict())!=digest(replace(problem,references=p.references,validation_status='reference_passed').to_dict())):
                    raise ValueError('accepted_problem_or_reference_changed')
                shard_accepted.append(p.to_dict())
        if len(shard_accepted)!=status['reference_passed'] or status['processed']!=len(problems):
            raise ValueError('shard_status_count_mismatch')
        accepted.extend(shard_accepted);statuses.append(status)
        receipts.append({'run':shard['run'],'status_sha256':file_digest(status_path),
            'run_manifest_sha256':file_digest(run/'run_manifest.json'),
            'record_sha256':{p.name:file_digest(p) for p in record_files}})
    if len(set(identities))!=1:raise ValueError('mixed_executor_or_scoring_code_in_pool')
    physical=[s['physical_executions'] for s in statuses]
    report={'completed_at':utc_now(),'complete':True,'evidence_level':'isolated_execution',
        'input_problems':manifest['input_problems'],'reference_passed':len(accepted),
        'quarantined':manifest['input_problems']-len(accepted),
        'physical_executions':sum(physical) if all(n is not None for n in physical) else None,
        'known_physical_executions':sum(s.get('known_physical_executions',s['physical_executions'] or 0) for s in statuses),
        'unknown_physical_attempts':sum(s.get('unknown_physical_attempts',0) for s in statuses),
        'pool_manifest_sha256':file_digest(output/'pool.json'),'shards':receipts,
        'executor_version':identities[0][0],'scoring_code_sha256':identities[0][1],
        'model_generations':0,'model_updates':0,'final_benchmark_tasks':0}
    write_jsonl(output/'accepted.jsonl',sorted(accepted,key=lambda p:p['problem_id']))
    report['accepted_sha256']=file_digest(output/'accepted.jsonl')
    write_json(output/'report.json',report)
    return report


def execute(output,image,attestation,max_wall_hours):
    if sys.platform!='linux':raise RuntimeError('Linux_execution_host_required')
    import fcntl
    # Process death releases the OS lock. Persistent files alone cannot distinguish
    # an active controller from a crashed one, and would block safe resume.
    with (output/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('reference_pool_controller_already_running')
        return _execute_locked(output,image,attestation,max_wall_hours)


def _execute_locked(output,image,attestation,max_wall_hours):
    from posttrain.execution.contracts import Limits
    from posttrain.execution.docker_backend import DockerBackend
    if not 0<max_wall_hours<=22:raise ValueError('CPU_job_wall_cap_must_be_at_most_22_hours')
    DockerBackend(image,Limits(time_s=30),attestation)  # Linux and all live gates before dispatch.
    manifest=checked_manifest(output)
    session=output/'sessions'/uuid.uuid4().hex;session.mkdir(parents=True)
    processes=[];logs=[];deadline=time.monotonic()+max_wall_hours*3600
    try:
        for shard in manifest['shards']:
            run=local_path(output,shard['run'])
            if (run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('complete'):continue
            command=[sys.executable,'-m','posttrain','validate-references','--image',image,
                '--attestation',str(attestation.resolve()),'--input',str(local_path(output,shard['input'])),
                '--output',str(run),'--max-references',str(manifest['max_references'])]
            if run.exists():command.append('--resume')
            log=(session/(run.name+'.log')).open('w',encoding='utf-8');logs.append(log)
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('reference_shard_failed')
            if time.monotonic()>=deadline:raise TimeoutError('CPU_reference_pool_wall_limit')
            time.sleep(1)
        if any(p.returncode for p in processes):raise RuntimeError('reference_shard_failed')
    finally:
        for p in processes:
            if p.poll() is None:p.send_signal(signal.SIGINT)
        forced=False
        for p in processes:
            if p.poll() is None:
                try:p.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait(timeout=10);forced=True
        for log in logs:log.close()
        write_json(session/'controller.json',{'finished_at':utc_now(),'return_codes':[p.returncode for p in processes],
            'forced_child_kill':forced,'container_cleanup_requires_verification':forced,
            'provider_billing_stopped':False})
    return collect(output)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--input',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True);prep.add_argument('--workers',type=int,default=4)
    prep.add_argument('--max-references',type=int,default=3)
    run=sub.add_parser('execute');run.add_argument('--output',type=Path,required=True)
    run.add_argument('--image',required=True);run.add_argument('--attestation',type=Path,required=True)
    run.add_argument('--max-wall-hours',type=float,default=22)
    merge=sub.add_parser('collect');merge.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.input,args.output,workers=args.workers,max_references=args.max_references)
    elif args.action=='execute':result=execute(args.output,args.image,args.attestation,args.max_wall_hours)
    else:result=collect(args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
