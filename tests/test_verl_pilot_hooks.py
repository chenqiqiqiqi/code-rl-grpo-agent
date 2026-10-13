from types import SimpleNamespace
import importlib
import importlib.util

import pytest

from posttrain.training.verl_pilot_trace import PilotTrace


def implementation():
    name='posttrain.training.verl_pilot_hooks'
    assert importlib.util.find_spec(name) is not None,'verl pilot hooks not implemented'
    return importlib.import_module(name)


class Batch:
    def __init__(self,batch=None,non_tensor=None,meta=None):
        self.batch=batch or {};self.non_tensor_batch=non_tensor or {};self.meta_info=meta or {}
    def __len__(self):
        values=self.batch or self.non_tensor_batch
        return len(next(iter(values.values())))


def generated():
    return Batch({'responses':[[4,5,0],[6,0,0]],'response_mask':[[1,1,0],[1,0,0]],
        'prompts':[[0,1,2],[0,1,2]],'attention_mask':[[0,1,1,1,1,0],[0,1,1,1,0,0]],
        'rollout_log_probs':[[-.2,-.4,0],[-.3,0,0]]},
        {'request_id':['0'*32,'1'*32],'binary':[0.,1.],'mixed':[.25,1.],
         'queue_s':[.1,.2],'judging_s':[.5,.6]},
        {'timing':{'agent_loop/generate_sequences/max':.7}})


def test_hooks_reserve_before_dispatch_and_keep_real_logprob_sources(tmp_path):
    module=implementation()
    identity={'scope':'engineering_pilot_only','run_uuid':'a'*32,'config_sha256':'b'*64}
    with PilotTrace(tmp_path/'audit',identity,100) as trace:
        class UpstreamFixture:
            def __init__(self):self.global_steps=1
            def init_workers(self):
                def generate(batch):
                    assert trace.snapshot()['tokens']['charged_tokens']==6
                    return generated()
                self.async_rollout_manager=SimpleNamespace(generate_sequences=generate)
                self.checkpoint_manager=SimpleNamespace(update_weights=lambda step:None)
            def _compute_old_log_prob(self,batch):
                return Batch({'old_log_probs':[[-.21,-.41,0],[-.31,0,0]]}),0.
            def _update_actor(self,batch):return Batch(meta={'metrics':{'actor/grad_norm':.1}})
        cls=module.instrumented_trainer(UpstreamFixture)
        trainer=cls(pilot_trace=trace,pilot_settings={'g':2,'max_response_tokens':3,'loss_agg_mode':'token-mean'})
        trainer.init_workers()
        source=Batch(non_tensor={'uid':['group','group'],
            'extra_info':[{'problem_id':'p'},{'problem_id':'p'}]})
        output=trainer.async_rollout_manager.generate_sequences(source)
        trainer._compute_old_log_prob(output)
        trainer._update_actor(output)
        trainer.checkpoint_manager.update_weights(1)
        snapshot=trace.snapshot()
        assert snapshot['tokens']['charged_tokens']==3
        assert len(snapshot['complete_update_and_sync_steps'])==1
        assert snapshot['real_7b_pilot_complete'] is False
        event=next(e for e in snapshot['events'] if e['phase']=='actor_update')
        assert event['payload']['active_response_tokens']==3
        assert event['payload']['active_sequences']==2
        assert list((trace.root/'old_log_probs').glob('*.json'))


def test_generation_failure_never_updates_or_refunds(tmp_path):
    module=implementation()
    identity={'scope':'engineering_pilot_only','run_uuid':'a'*32,'config_sha256':'b'*64}
    with PilotTrace(tmp_path/'audit',identity,100) as trace:
        class UpstreamFixture:
            def __init__(self):self.global_steps=1
            def init_workers(self):
                def fail(batch):raise RuntimeError('synthetic transport failure')
                self.async_rollout_manager=SimpleNamespace(generate_sequences=fail)
                self.checkpoint_manager=SimpleNamespace(update_weights=lambda step:None)
        trainer=module.instrumented_trainer(UpstreamFixture)(pilot_trace=trace,
            pilot_settings={'g':2,'max_response_tokens':3,'loss_agg_mode':'token-mean'})
        trainer.init_workers()
        with pytest.raises(RuntimeError,match='synthetic'):
            trainer.async_rollout_manager.generate_sequences(Batch(non_tensor={'uid':['g','g']}))
        snapshot=trace.snapshot()
        assert snapshot['tokens']['charged_tokens']==6
        assert snapshot['completed_updates']==0
        assert snapshot['tokens']['reservations'][0]['actual'] is None


def test_sync_return_alone_is_not_a_weight_identity_probe(tmp_path):
    module=implementation()
    identity={'scope':'engineering_pilot_only','run_uuid':'a'*32,'config_sha256':'b'*64}
    with PilotTrace(tmp_path/'audit',identity,100) as trace:
        class UpstreamFixture:
            def __init__(self):self.global_steps=0
            def init_workers(self):
                self.async_rollout_manager=SimpleNamespace()
                self.checkpoint_manager=SimpleNamespace(update_weights=lambda step:None)
        trainer=module.instrumented_trainer(UpstreamFixture)(pilot_trace=trace,
            pilot_settings={'g':2,'max_response_tokens':3,'loss_agg_mode':'token-mean'})
        trainer.init_workers();trainer.checkpoint_manager.update_weights(0)
        sync=next(e for e in trace.snapshot()['events'] if e['phase']=='sync')
        assert sync['payload']['fixed_token_weight_probe_passed'] is False
        assert trace.snapshot()['complete_update_and_sync_steps']==[]
