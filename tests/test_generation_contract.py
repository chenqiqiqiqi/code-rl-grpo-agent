from dataclasses import replace
import pytest

from posttrain.data.prompt_manifest import public_prompt,checked_prompt
from posttrain.training.generate import checked_generation,validate_generated


def test_explicit_repetition_penalty_is_validated_and_forwarded():
    import json
    from pathlib import Path
    from posttrain.training.generate import sampling_arguments
    cfg=json.loads(Path('configs/sft_development_greedy_candidate.json').read_text())
    assert sampling_arguments(checked_generation(cfg))['repetition_penalty']==1.0
    assert sampling_arguments(checked_generation({**cfg,'repetition_penalty':1.1}))['repetition_penalty']==1.1
    for value in (0,-1,float('nan'),True,'1.1'):
        with pytest.raises(ValueError,match='repetition_penalty'):
            checked_generation({**cfg,'repetition_penalty':value})


def test_gpu_prompt_export_whitelists_fields_and_rejects_final_or_unverified(problem):
    p=replace(problem,validation_status='reference_passed',cluster_id='c')
    row=public_prompt(p,'development')
    assert 'tests' not in row and 'references' not in row
    assert row['messages']==p.prompt()
    with pytest.raises(ValueError):public_prompt(p,'final')
    with pytest.raises(ValueError):public_prompt(problem,'pilot')
    with pytest.raises(ValueError):checked_prompt({**row,'references':list(p.references)})


def test_generation_accounting_rejects_missing_reordered_or_unlogged_outputs():
    prompts=[{'problem_id':'a'},{'problem_id':'b'}]
    rows=[{'problem_id':p['problem_id'],'candidate_index':j,'token_ids':[1,2],
        'output_tokens':2,'sampled_token_log_probs':[-.1,-.2]} for p in prompts for j in range(2)]
    validate_generated(rows,prompts,2,4)
    with pytest.raises(ValueError):validate_generated(rows[:-1],prompts,2,4)
    with pytest.raises(ValueError):validate_generated(list(reversed(rows)),prompts,2,4)
    with pytest.raises(ValueError):validate_generated([{**rows[0],'output_tokens':3}]+rows[1:],prompts,2,4)
    with pytest.raises(ValueError):validate_generated([{**rows[0],'sampled_token_log_probs':[]}]+rows[1:],prompts,2,4)


def test_generation_bundle_rejects_replaced_candidates_and_changed_ledger(tmp_path):
    # Contract fixture only: no GPU is loaded and no model output is produced.
    import sqlite3
    from posttrain.common import digest,file_digest,write_json,write_jsonl
    from posttrain.training.budget import TokenLedger
    from posttrain.training.generate import checked_generation_bundle
    prompts=[{'problem_id':'a'}];run='fixture-run'
    ledger=TokenLedger(tmp_path/'token-ledger.sqlite',4)
    reservation=ledger.reserve(1,2,2,'offline_preparation_generation')
    rows=[{'problem_id':'a','candidate_index':j,'token_ids':[1,2],'output_tokens':2,
        'sampled_token_log_probs':[-.1,-.2],'request_id':digest([run,0,j])[:32],'reservation':reservation} for j in range(2)]
    ledger.settle(reservation,4)
    write_json(tmp_path/'identity.json',{'run_uuid':run,'configuration':{'g':2,'max_new_tokens':2,'token_cap':4},
        'prompts_sha256':'frozen-prompts'})
    write_jsonl(tmp_path/'candidates.jsonl',rows)
    report={'evidence_level':'real_7b_offline_generation','complete':True,
        'identity_sha256':file_digest(tmp_path/'identity.json'),
        'candidates_sha256':file_digest(tmp_path/'candidates.jsonl'),'token_accounting':ledger.snapshot()}
    write_json(tmp_path/'report.json',report);ledger.close()
    checked_generation_bundle(tmp_path,prompts,'frozen-prompts')
    write_jsonl(tmp_path/'candidates.jsonl',[{**rows[0],'output_tokens':1},rows[1]])
    with pytest.raises(ValueError,match='artifact_identity'):checked_generation_bundle(tmp_path,prompts,'frozen-prompts')
    write_jsonl(tmp_path/'candidates.jsonl',rows)
    with sqlite3.connect(tmp_path/'token-ledger.sqlite') as db:db.execute('UPDATE reservations SET actual=3')
    with pytest.raises(ValueError,match='ledger_changed'):checked_generation_bundle(tmp_path,prompts,'frozen-prompts')
