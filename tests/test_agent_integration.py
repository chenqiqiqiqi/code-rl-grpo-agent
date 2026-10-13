"""Synthetic transport/API contracts; these tests never execute candidate programs."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType,SimpleNamespace

import pytest
from posttrain.agent.tool_service import make_server,ToolClient,ToolInfrastructureError
from posttrain.execution.contracts import CaseResult,Status
from posttrain.training.code_rl import checked_settings,check_prepared,validate_v3_freeze
from posttrain.training.sft_input import load_example


def test_tool_only_supplied_input_and_at_most_once(tmp_path):
    calls=[]
    def run(code,stdin,*args):
        calls.append((code,stdin))
        return CaseResult(Status.OK,stdout='42\n',stderr='',exit_code=0)
    backend=SimpleNamespace(evidence_level='isolated_execution',version='synthetic',run=run)
    server=make_server(backend,tmp_path,port=0,workers=1)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=ToolClient(f'http://127.0.0.1:{server.server_port}','synthetic')
        result=client.run('NOT EXECUTED','supplied input','a'*32)
        assert result['stdout']=='42\n'
        assert client.run('NOT EXECUTED','supplied input','a'*32)==result
        assert calls==[('NOT EXECUTED','supplied input')]
        with pytest.raises(ToolInfrastructureError):client.run('DIFFERENT','supplied input','a'*32)
        assert len(calls)==1
    finally:
        server.shutdown();server.server_close();thread.join()


def test_sft_gpu_row_rejects_hidden_fields():
    row=dict(schema='public_sft_v1',problem_id='p',cluster_id='c',messages=[{'role':'system','content':'s'},{'role':'user','content':'u'}],answer='print(1)',source_row_sha256='a'*64)
    example=load_example(row)
    assert example.references==('print(1)',)
    assert not hasattr(example,'tests')
    with pytest.raises(ValueError):load_example({**row,'tests':[]})


@pytest.mark.parametrize('final_text,expected_code,submission_format',[
    (json.dumps({'action':'submit','code':'print(42)'}),'print(42)','json_action'),
    ('Solution:\n```python\nprint(42)\n```','print(42)\n','python_fence_fallback'),
])
def test_agent_mask_and_terminal_only_scoring(monkeypatch,tmp_path,final_text,expected_code,submission_format):
    stub=ModuleType('verl.experimental.agent_loop.agent_loop')
    stub.AgentLoopBase=object
    stub.AgentLoopOutput=lambda **kw:SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules,'verl.experimental.agent_loop.agent_loop',stub)
    spec=importlib.util.spec_from_file_location('agent_loop_fixture',Path('src/posttrain/agent/verl_loop.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actions={10:json.dumps({'action':'run_python','code':'unexecuted','stdin':'public'}),
             20:final_text}
    loop=module.CodeAgentLoop.__new__(module.CodeAgentLoop)
    messages=[{'role':'system','content':'s'},{'role':'user','content':'p'}]
    loop.context={'audit_directory':str(tmp_path),'reward_kind':'binary'}
    loop.context_sha256='a'*64;loop.prompts={'p':{'problem_id':'p','messages':messages}}
    loop.scorer=object();loop.max_tool_calls=3;loop.max_generated_tokens=8;loop.max_turn_tokens=4
    loop.rollout_config=SimpleNamespace(response_length=20)
    loop.tokenizer=SimpleNamespace(decode=lambda ids,**kw:actions[ids[0]])
    calls=[]
    async def initial(messages):return [1,2]
    async def generate(**kw):
        ids=[10] if len(calls)==0 else [20]
        calls.append(kw)
        return SimpleNamespace(token_ids=ids,log_probs=[-.5])
    async def merge(tokens,ids,mask,probs,assistant_logprobs):
        return SimpleNamespace(token_ids=tokens+ids),mask+[1]*len(ids),probs+assistant_logprobs
    async def context(previous,messages,tokens,mask,probs):
        return SimpleNamespace(token_ids=tokens+[30,31]),mask+[0,0],probs+[0.,0.]
    loop.ct_build_initial_tokens=initial;loop.ct_merge_assistant_token=merge;loop.ct_merge_context_msg=context
    loop.server_manager=SimpleNamespace(generate=generate)
    loop.tool=SimpleNamespace(run=lambda *args:{'stdout':'public output'})
    scored=[]
    def score(code,*args):
        scored.append(code)
        return {'score':1.,'binary':1.,'mixed':1.,'request_id':args[3],'queue_s':0.,'judging_s':0.}
    monkeypatch.setattr(module,'score_candidate',score)
    result=asyncio.run(loop.run({},raw_prompt=messages,extra_info={'problem_id':'p'}))
    assert result.response_ids==[10,30,31,20]
    assert result.response_mask==[1,0,0,1]
    assert result.response_logprobs==[-.5,0.,0.,-.5]
    assert scored==[expected_code]
    assert result.reward_score==1.
    assert result.extra_fields['reward_extra_info']['submission_format']==submission_format
    trajectory=json.loads(next((tmp_path/'episodes').glob('*.json')).read_text())
    assert trajectory['submission_format']==submission_format
    assert trajectory['action_events'][-1]['source']==submission_format
    assert trajectory['turns'][-1]['text_sha256']==hashlib.sha256(final_text.encode()).hexdigest()
    assert len(calls)==2


def test_settings_and_formal_gate_cannot_accept_configuration_as_evidence(tmp_path):
    cfg=json.loads(Path('configs/agent_rl_pilot.json').read_text())
    checked_settings(cfg)
    with pytest.raises(ValueError):checked_settings({**cfg,'g':1})
    with pytest.raises(ValueError):checked_settings({**cfg,'max_generated_tokens':99999})
    with pytest.raises(ValueError):validate_v3_freeze({'protocol':'code_agent_rl_v3','seeds':[42],'pilot_evidence':{'evidence_level':'static'}},tmp_path,cfg,tmp_path/'missing')


def test_formal_budget_end_restores_last_completed_data_cursor(tmp_path):
    from posttrain.training.verl_pilot_hooks import instrumented_trainer
    from posttrain.training.budget import BudgetExceeded
    from posttrain.common import write_json
    restored=[]
    trace=SimpleNamespace(root=tmp_path,session='session',ledger=SimpleNamespace(snapshot=lambda:{'remaining':3}),snapshot=lambda:{'fixture':True})
    class Base:
        def __init__(self):
            self.global_steps=3
            self._last_synced_step=2
            self._last_complete_data_state={'batch':2}
            self.train_dataloader=SimpleNamespace(load_state_dict=restored.append)
            self.config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path/'checkpoints')))
        def fit(self):raise BudgetExceeded('fixture_next_group_does_not_fit')
    trainer=instrumented_trainer(Base)(pilot_trace=trace,pilot_settings={'scope':'code_rl_v3'})
    saved=[];trainer._save_checkpoint=lambda:saved.append(trainer.global_steps)
    trainer.fit()
    assert restored==[{'batch':2}] and saved==[2]
    assert json.loads((tmp_path/'completion.json').read_text())['last_complete_update']==2
    assert trainer.global_steps==3


def test_infra_tool_failure_does_not_get_a_numeric_score(tmp_path):
    backend=SimpleNamespace(evidence_level='isolated_execution',version='synthetic',
                            run=lambda *args:CaseResult(Status.INFRA,physical_executions=None))
    server=make_server(backend,tmp_path,port=0,workers=1)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=ToolClient(f'http://127.0.0.1:{server.server_port}','synthetic')
        with pytest.raises(ToolInfrastructureError):client.run('NOT EXECUTED','','f'*32)
    finally:server.shutdown();server.server_close();thread.join()


def test_context_identity_detects_tool_mutation_but_ignores_padding():
    from posttrain.training.verl_pilot_hooks import sequence_identities
    batch=SimpleNamespace(batch={'prompts':[[0,1]],'responses':[[2,3,4,0]],
        'attention_mask':[[0,1,1,1,1,0]],'response_mask':[[1,0,1,0]]},
        non_tensor_batch={'request_id':['r']})
    before=sequence_identities(batch)
    batch.batch['responses'][0][-1]=999
    assert sequence_identities(batch)==before
    batch.batch['responses'][0][1]=88
    assert sequence_identities(batch)!=before


def test_prepared_manifest_cannot_omit_data_hashes(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'hidden_tests_exported':False,'files':{}}))
    with pytest.raises(ValueError,match='missing_required_files'):check_prepared(tmp_path,'agent')
