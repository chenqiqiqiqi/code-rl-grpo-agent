"""Contract fixtures only: no GPU, model update or live scoring occurs here."""
import importlib
import importlib.util
import json

import pytest

from posttrain.training.budget import BudgetExceeded


def implementation():
    name='posttrain.training.verl_pilot_trace'
    assert importlib.util.find_spec(name) is not None,'pilot trace not implemented'
    return importlib.import_module(name)


def identity():
    return {'scope':'engineering_pilot_only','run_uuid':'a'*32,'config_sha256':'b'*64}


def candidates():
    return [{'uid':'group','problem_id':'p','request_id':str(i)*32,
             'prompt_tokens':[10,11], 'response_tokens':[12,13],
             'rollout_log_probs':[-.2,-.3], 'binary':i,'mixed':float(i),
             'queue_s':.1,'judging_s':.2} for i in (0,1)]


def test_resumed_pilot_retains_unknown_rollout_charge_and_identity(tmp_path):
    module=implementation()
    run=tmp_path/'run'
    with module.PilotTrace(run,identity(),16) as trace:
        trace.begin_rollout(1,1,2,4)
        assert trace.snapshot()['tokens']['charged_tokens']==8
    with module.PilotTrace(run,identity(),16,resume=True) as trace:
        second=trace.begin_rollout(1,1,2,4)
        trace.finish_rollout(second,candidates(),{'agent_generation_max_s':.5},.8)
        assert trace.snapshot()['tokens']['charged_tokens']==12
        with pytest.raises(BudgetExceeded):trace.begin_rollout(2,1,2,4)
        assert trace.snapshot()['completed_updates']==0
    with pytest.raises(ValueError,match='identity'):
        module.PilotTrace(run,{**identity(),'config_sha256':'c'*64},16,resume=True)


def test_failed_sync_is_not_a_complete_step_or_pilot_acceptance(tmp_path):
    module=implementation()
    with module.PilotTrace(tmp_path/'run',identity(),100) as trace:
        reservation=trace.begin_rollout(1,1,2,4)
        trace.finish_rollout(reservation,candidates(),{},.8)
        trace.phase(1,'old_log_prob',.2,{'source':'actor_recomputed'})
        trace.phase(1,'actor_update',.3,{'response_tokens':4,'loss_agg_mode':'token-mean'})
        trace.phase(1,'sync',.1,{'failure':'fixture'},complete=False)
        assert trace.snapshot()['completed_updates']==1
        assert trace.snapshot()['complete_update_and_sync_steps']==[]
        assert trace.snapshot()['real_7b_pilot_complete'] is False


def test_partial_candidate_group_does_not_settle_budget(tmp_path):
    module=implementation()
    with module.PilotTrace(tmp_path/'run',identity(),100) as trace:
        rid=trace.begin_rollout(1,1,2,4)
        with pytest.raises(ValueError,match='candidate_count'):
            trace.finish_rollout(rid,candidates()[:1],{},.2)
        assert trace.snapshot()['tokens']['charged_tokens']==8
        assert trace.snapshot()['tokens']['reservations'][0]['actual'] is None


def test_old_log_prob_audit_preserves_tokens_and_rejects_nonfinite(tmp_path):
    module=implementation()
    with module.PilotTrace(tmp_path/'run',identity(),100) as trace:
        rid=trace.begin_rollout(1,1,2,4)
        trace.finish_rollout(rid,candidates(),{},.2)
        rows=[{'request_id':'0'*32,'tokens':[12,13],'old_log_probs':[-.25,-.4]}]
        path=trace.log_probs(1,rows)
        saved=json.loads(path.read_text(encoding='utf-8'))
        assert saved['rows']==rows
        assert saved['source']=='actor_recomputed_before_update'
        with pytest.raises(ValueError,match='nonfinite'):
            trace.log_probs(2,[{**rows[0],'old_log_probs':[-.2,float('nan')]}])


def test_checkpoint_receipt_rejects_missing_or_modified_resume_state(tmp_path):
    module=implementation()
    checkpoint=tmp_path/'checkpoint'; checkpoint.mkdir()
    (checkpoint/'data.pt').write_bytes(b'fixture data progress')
    actor=checkpoint/'actor';actor.mkdir()
    (actor/'model_world_size_1_rank_0.pt').write_bytes(b'fixture model')
    with pytest.raises(ValueError,match='optimizer'):
        module.checkpoint_receipt(checkpoint)
    (actor/'optim_world_size_1_rank_0.pt').write_bytes(b'fixture optimizer')
    (actor/'extra_state_world_size_1_rank_0.pt').write_bytes(b'fixture scheduler and RNG')
    receipt=module.checkpoint_receipt(checkpoint)
    module.verify_checkpoint(checkpoint,receipt)
    (checkpoint/'data.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint_changed'):
        module.verify_checkpoint(checkpoint,receipt)
