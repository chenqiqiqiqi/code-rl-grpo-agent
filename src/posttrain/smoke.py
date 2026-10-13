from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from posttrain.common import digest,utc_now,write_json,write_jsonl
from posttrain.data.dedup import cluster_problems
from posttrain.data.sampling import split_clusters
from posttrain.data.schema import Problem,TestCase
from posttrain.execution.contracts import CaseResult,Limits,Status
from posttrain.execution.judge import Judge
from posttrain.execution.replay import ReplayBackend
from posttrain.rewards import group_statistics,rewards


def contract_smoke(output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    problems=[Problem(f'synthetic-{i:03d}','synthetic','v1',f'synthetic:{i}',
        f'Read an integer n and print n plus {i}.',tuple(TestCase(f'{n}\n',f'{n+i}\n') for n in range(5)),
        (f'print(int(input())+{i})',),difficulty='synthetic') for i in range(100)]
    problems,links=cluster_problems(problems)
    split=split_clusters(problems)
    rows,groups=[],[]
    for p in problems:
        group=[]
        for candidate in range(4):
            # This backend returns scripted outcomes without executing either code string.
            outcomes=[CaseResult(Status.OK,stdout=t.stdout if candidate==0 or j<candidate else 'WRONG',queue_s=0,physical_executions=0)
                      for j,t in enumerate(p.tests)]
            backend=ReplayBackend(outcomes)
            judge=Judge(backend)
            result=judge.score(p.references[0]+f'\n# candidate {candidate}',p.tests,Limits())
            group.append(result)
            rows.append({'problem_id':p.problem_id,'candidate':candidate,'result':result.to_dict(),'reward':rewards(result)})
        groups.append(group)
    stats=group_statistics(groups,4)
    write_jsonl(output/'scores.jsonl',rows)
    write_json(output/'split.json',split)
    write_json(output/'dedup.json',links)
    summary={'run_id':output.name,'created_at':utc_now(),'evidence_level':'mock_contract',
             'description':'100 synthetic problems, scripted outputs; no candidate/reference execution or model generation',
             'problems':len(problems),'candidates':len(rows),'case_judgments':sum(x['result']['total'] for x in rows),
             'physical_candidate_executions':0,'model_updates':0,'external_spend_cny':0,
             'elapsed_s':time.monotonic()-started,'group_statistics':stats,'score_file_sha256':digest(rows),
             'stage1_live_acceptance_passed':False}
    write_json(output/'summary.json',summary)
    return summary
