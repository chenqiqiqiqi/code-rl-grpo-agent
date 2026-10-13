"""Resumable isolated reference validation, with immutable per-problem attempts."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from posttrain.common import digest,file_digest,read_jsonl,utc_now,write_json,write_jsonl
from posttrain.data.prepare import validate_references,ReferenceValidationError
from posttrain.data.schema import Problem
from posttrain.execution.judge import Judge


def reference_run(source: Path, output: Path, backend, *, resume=False, max_references=3) -> dict:
    if backend.evidence_level!='isolated_execution' or type(max_references) is not int or max_references<1:
        raise ValueError('isolated_backend_and_positive_reference_budget_required')
    root=Path(__file__).resolve().parents[1]
    identity={'source_sha256':file_digest(source),'executor_version':backend.version,
              'max_references':max_references,
              'scoring_code_sha256':digest({str(p.relative_to(root)):file_digest(p) for folder in ('data','execution')
                                           for p in sorted((root/folder).glob('*.py'))})}
    if resume:
        manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
        if manifest['identity']!=identity:
            raise ValueError('resume_input_executor_or_code_changed_start_new_run')
    else:
        output.mkdir(parents=True,exist_ok=False)
        manifest={'run_id':output.name,'created_at':utc_now(),'identity':identity,
                  'evidence_level':'isolated_execution','reference_selection':'first_N_in_source_order'}
        write_json(output/'run_manifest.json',manifest)
    records=output/'records';records.mkdir(exist_ok=True)
    intents=output/'intents';intents.mkdir(exist_ok=True)
    # A process kill can bypass finally. Preserve every dispatch intent even if
    # its result was never written; a retry must not turn unknown work into zero.
    for intent_path in sorted(intents.glob('*.json')):
        record_path=records/intent_path.name
        if not record_path.exists():
            abandoned=json.loads(intent_path.read_text(encoding='utf-8'))
            abandoned.update(interruption='unknown_after_restart',physical_accounting_complete=False)
            write_json(record_path,abandoned)
    problems=[Problem.from_dict(row) for row in read_jsonl(source)]
    if len({p.problem_id for p in problems})!=len(problems):
        raise ValueError('duplicate_reference_input_ids')
    accepted,audit=[],[]
    finished={}
    for path in sorted(records.glob('*.json')):
        entry=json.loads(path.read_text(encoding='utf-8'))
        audit.extend(entry['audit'])
        if entry['complete']:
            finished[entry['problem_id']]=entry
    for p in problems:
        if p.problem_id in finished and finished[p.problem_id]['accepted']:
            accepted.append(Problem.from_dict(finished[p.problem_id]['accepted']))
    started=time.monotonic()
    try:
        for index,p in enumerate(problems):
            if p.problem_id in finished:
                continue
            attempts=sorted(records.glob(f'{index:05d}-*.json'))
            attempt=len(attempts)+1
            entry={'problem_id':p.problem_id,'attempt':attempt,'complete':False,'accepted':None,
                   'available_references':len(p.references),'reference_budget':max_references,'audit':[],
                   'started_at':utc_now(),'physical_accounting_complete':False}
            write_json(intents/f'{index:05d}-{attempt:03d}.json',entry)
            try:
                good,rows=validate_references([replace(p,references=p.references[:max_references])],Judge(backend))
                if not good:
                    rows[-1]['quarantine_reason']='no_passing_reference_within_registered_budget'
                entry.update(complete=True,accepted=good[0].to_dict() if good else None,audit=rows,
                             physical_accounting_complete=True)
                accepted.extend(good)
            except ReferenceValidationError as exc:
                entry['audit']=exc.audit
                entry['physical_accounting_complete']=True
                raise
            finally:
                # Immutable attempt records are the source of truth after a crash.
                write_json(records/f'{index:05d}-{attempt:03d}.json',entry)
                audit.extend(entry['audit'])
                write_jsonl(output/'reference_audit.jsonl',audit)
                write_jsonl(output/'accepted.jsonl',(x.to_dict() for x in accepted))
            finished[p.problem_id]=entry
            state={'run_id':output.name,'complete':False,'processed':len(finished),'input_problems':len(problems),
                   'reference_passed':len(accepted),'updated_at':utc_now()}
            write_json(output/'status.json',state)
            print(json.dumps(state),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',{'run_id':output.name,'complete':False,'error':type(exc).__name__,
                   'processed':len(finished),'reference_passed':len(accepted),'updated_at':utc_now()})
        raise
    physical=[r['result']['physical_executions'] for r in audit if 'result' in r]
    all_attempts=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(records.glob('*.json'))]
    unknown_attempts=sum(not entry.get('physical_accounting_complete',entry['complete']) for entry in all_attempts)
    known_physical=sum(x for x in physical if x is not None)
    result={'run_id':output.name,'evidence_level':'isolated_execution','input_problems':len(problems),
            'processed':len(finished),'reference_passed':len(accepted),'quarantined':len(problems)-len(accepted),
            'complete':True,'updated_at':utc_now(),'current_session_wall_s':time.monotonic()-started,
            'physical_executions':known_physical if not unknown_attempts and all(x is not None for x in physical) else None,
            'known_physical_executions':known_physical,'unknown_physical_attempts':unknown_attempts,
            'attempts_including_resumed_work':len(all_attempts)}
    write_json(output/'status.json',result)
    return result
