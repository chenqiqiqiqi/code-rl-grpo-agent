from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from posttrain.common import write_jsonl, write_json
import posttrain.data.reference_run as reference_module


def test_jsonl_failed_rewrite_keeps_last_complete_artifact(tmp_path):
    path=tmp_path/'rows.jsonl'
    write_jsonl(path,[{'before':1}])
    original=path.read_bytes()
    def broken():
        yield {'partial':1}
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        write_jsonl(path,broken())
    assert path.read_bytes()==original


def test_reference_dispatch_has_durable_intent(tmp_path, problem, monkeypatch):
    # Mock control-flow check only: no reference code is executed by this test.
    source=tmp_path/'input.jsonl'; output=tmp_path/'run'
    write_jsonl(source,[problem.to_dict()])
    backend=SimpleNamespace(evidence_level='isolated_execution',version='unit-test-only')
    def validate(problems, judge):
        intent=json.loads((output/'intents/00000-001.json').read_text())
        assert intent['problem_id']==problem.problem_id
        return [],[{'problem_id':problem.problem_id,'quarantine_reason':'synthetic_test'}]
    monkeypatch.setattr(reference_module,'validate_references',validate)
    result=reference_module.reference_run(source,output,backend)
    assert result['physical_executions']==0


def test_reference_resume_preserves_unknown_physical_work(tmp_path, problem, monkeypatch):
    source=tmp_path/'input.jsonl'; output=tmp_path/'run'
    write_jsonl(source,[problem.to_dict()])
    backend=SimpleNamespace(evidence_level='isolated_execution',version='unit-test-only')
    def interrupted(*args):
        raise KeyboardInterrupt('simulated lost in-flight result')
    monkeypatch.setattr(reference_module,'validate_references',interrupted)
    with pytest.raises(KeyboardInterrupt):
        reference_module.reference_run(source,output,backend)
    # Simulate SIGKILL after durable intent but before a completion record.
    (output/'records/00000-001.json').unlink()
    def finish(problems, judge):
        return [replace(problem,validation_status='reference_passed')],[{
            'problem_id':problem.problem_id,'result':{'physical_executions':5}}]
    monkeypatch.setattr(reference_module,'validate_references',finish)
    result=reference_module.reference_run(source,output,backend,resume=True)
    assert result['complete'] and result['reference_passed']==1
    assert result['attempts_including_resumed_work']==2
    assert result['physical_executions'] is None
    assert result['known_physical_executions']==5
    assert result['unknown_physical_attempts']==1
    orphan=json.loads((output/'records/00000-001.json').read_text())
    assert orphan['physical_accounting_complete'] is False
    assert orphan['interruption']=='unknown_after_restart'
