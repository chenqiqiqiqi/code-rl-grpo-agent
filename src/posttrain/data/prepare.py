from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from posttrain.common import digest, file_digest, utc_now, write_json, write_jsonl
from posttrain.data.dedup import cluster_problems
from posttrain.data.sampling import split_clusters
from posttrain.data.schema import DataError, Problem
from posttrain.data.taco import from_taco
from posttrain.execution.contracts import Limits
from posttrain.execution.judge import Judge


class ReferenceValidationError(RuntimeError):
    def __init__(self,problem_id: str,audit: list[dict]):
        super().__init__(f'reference_backtest_invalid:{problem_id}')
        self.audit=audit


def prepare_taco(path: Path, revision: str, output: Path, limit: int = 100) -> dict:
    import pyarrow.parquet as pq
    if limit < 2:
        raise ValueError('need at least two problems')
    problems, rejected, rejection_rows = [],Counter(),[]
    scanned = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            idx = scanned
            scanned += 1
            try:
                problems.append(from_taco(row,revision,idx))
            except DataError as e:
                rejected[str(e)] += 1
                rejection_rows.append({'row_index':idx,'reason':str(e)})
            if len(problems) == limit:
                break
        if len(problems) == limit:
            break
    problems,links = cluster_problems(problems)
    write_jsonl(output/'problems.jsonl',(p.to_dict() for p in problems))
    write_jsonl(output/'dedup_links.jsonl',links)
    write_jsonl(output/'rejections.jsonl',rejection_rows)
    splits = split_clusters(problems)
    splits['status'] = 'provisional_not_reference_validated_not_final_decontaminated'
    write_json(output/'provisional_split.json',splits)
    report = {'run_id':output.name,'created_at':utc_now(),'evidence_level':'real_data_static_validation',
              'source':str(path),'source_sha256':file_digest(path),'revision':revision,'scanned':scanned,
              'accepted':len(problems),'requested':limit,'rejected':dict(rejected),
              'clusters':len({p.cluster_id for p in problems}),'dedup_links':len(links),
              'tests':sum(len(p.tests) for p in problems),'reference_executions':0,
              'reference_validation_complete':False,'final_decontamination_complete':False,
              'problem_manifest_hash':digest([p.to_dict() for p in problems])}
    write_json(output/'prepare_report.json',report)
    return report


def validate_references(problems: list[Problem], judge: Judge) -> tuple[list[Problem],list[dict]]:
    if judge.backend.evidence_level != 'isolated_execution':
        raise ValueError('reference acceptance requires isolated_execution, not mock/replay')
    accepted,audit = [],[]
    for p in problems:
        for index,code in enumerate(p.references):
            result = judge.score(code,p.tests,Limits(time_s=p.time_limit_s))
            audit.append({'problem_id':p.problem_id,'reference_index':index,'result':result.to_dict()})
            if not result.valid:
                # Infrastructure failure is not evidence the reference is incorrect.
                raise ReferenceValidationError(p.problem_id,audit)
            if result.passed == result.total:
                accepted.append(replace(p,references=(code,),validation_status='reference_passed'))
                break
        else:
            audit.append({'problem_id':p.problem_id,'quarantine_reason':'all_references_failed'})
    return accepted,audit
