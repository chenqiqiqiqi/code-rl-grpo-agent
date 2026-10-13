"""Observation hooks for the pinned synchronous RayPPOTrainer.

No upstream source patching and no final-benchmark access. Returning from weight
transfer is recorded separately from proving fixed-token weight equivalence.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from posttrain.common import digest,file_digest,write_json
from posttrain.training.budget import BudgetExceeded
from posttrain.training.verl_pilot_trace import checkpoint_receipt,finite,verify_checkpoint


class GeneratedTokenTargetReached(BudgetExceeded):
    """Phase-2 equal-budget stop (B2): raised before the next rollout once the completed updates
    of this run's lineage have generated at least `stop_generated_tokens` model tokens."""


def lineage_generated_tokens(snapshot,before_step):
    """Model-generated tokens (tool observations are masked out) of the rollouts behind updates
    1..before_step-1. After a resume, steps past the checkpoint are re-run: only the latest settled
    rollout of each step counts, so abandoned or crashed attempts are never double-counted."""
    latest={}
    for row in snapshot['reservations']:
        purpose=row['payload']['purpose']
        if row['actual'] is None or not purpose.startswith('pilot:'):continue
        latest[int(purpose.rsplit(':',1)[1])]=row['actual']
    return sum(v for step,v in latest.items() if step<before_step)


def plain(value):
    if hasattr(value,'detach'):value=value.detach().cpu()
    if hasattr(value,'tolist'):value=value.tolist()
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    return value


def selected(values,mask):
    if len(values)!=len(mask) or any(v not in (0,1) for v in mask):
        raise ValueError('invalid_response_mask')
    return [v for v,m in zip(values,mask) if m]


def generated_rows(source,output):
    tensors=plain(dict(output.batch));metadata=plain(output.non_tensor_batch)
    inputs=plain(source.non_tensor_batch)
    rows=[]
    for i in range(len(source)):
        mask=tensors['response_mask'][i];prompt=tensors['prompts'][i]
        rows.append({'uid':inputs['uid'][i],'problem_id':inputs['extra_info'][i]['problem_id'],
            'request_id':metadata['request_id'][i],
            'prompt_tokens':selected(prompt,tensors['attention_mask'][i][:len(prompt)]),
            'response_tokens':selected(tensors['responses'][i],mask),
            'rollout_log_probs':selected(tensors['rollout_log_probs'][i],mask),
            **{k:metadata[k][i] for k in ('binary','queue_s','judging_s')},
            'mixed':metadata['mixed'][i] if 'mixed' in metadata else None,
            'mixed_available':metadata['mixed_available'][i] if 'mixed_available' in metadata else True,
            'scoring_mode':metadata['scoring_mode'][i] if 'scoring_mode' in metadata else 'full'})
    return rows


def sequence_identities(batch):
    """Bind tool context as well as generated tokens; padding is excluded."""
    tensors=plain(dict(batch.batch))
    result={}
    for i,rid in enumerate(plain(batch.non_tensor_batch['request_id'])):
        prompt=tensors['prompts'][i];response=tensors['responses'][i]
        attention=tensors['attention_mask'][i]
        active=attention[len(prompt):]
        result[rid]=digest({'tokens':selected(prompt+response,attention),
                           'response_mask':selected(tensors['response_mask'][i],active)})
    return result


class _Proxy:
    def __init__(self,target):self.target=target
    def __getattr__(self,name):return getattr(self.target,name)


class _ObservedRollout(_Proxy):
    def __init__(self,target,trainer):super().__init__(target);self.trainer=trainer

    def generate_sequences(self,batch):
        trainer=self.trainer;trace=trainer.pilot_trace;settings=trainer.pilot_settings
        if len(batch)%settings['g']:raise ValueError('pilot_requires_complete_candidate_groups')
        target=settings.get('stop_generated_tokens')
        if target and lineage_generated_tokens(trace.ledger.snapshot(),trainer.global_steps)>=target:
            raise GeneratedTokenTargetReached('generated_token_target_reached')
        rid=trace.begin_rollout(trainer.global_steps,len(batch)//settings['g'],
            settings['g'],settings.get('max_generated_tokens',settings['max_response_tokens']))
        started=time.monotonic()
        try:
            output=self.target.generate_sequences(batch)
            wall=time.monotonic()-started
            rows=generated_rows(batch,output)
            timing={k:plain(v) for k,v in output.meta_info['timing'].items()
                if k.startswith(('agent_loop/generate_sequences/','agent_loop/compute_score/','agent_loop/tool_calls/'))}
            trace.finish_rollout(rid,rows,timing,wall)
            trainer._pilot_generated={r['request_id']:r['response_tokens'] for r in rows}
            trainer._pilot_sequences=sequence_identities(output)
            return output
        except BaseException as exc:
            trace.phase(trainer.global_steps,'rollout_failed',time.monotonic()-started,
                {'failure_type':type(exc).__name__,'reservation_id':rid},complete=False)
            raise


class _ObservedCheckpointManager(_Proxy):
    def __init__(self,target,trainer):super().__init__(target);self.trainer=trainer

    def update_weights(self,step):
        started=time.monotonic();trace=self.trainer.pilot_trace
        try:
            result=self.target.update_weights(step)
        except BaseException as exc:
            trace.phase(step,'sync',time.monotonic()-started,
                {'failure_type':type(exc).__name__},complete=False)
            raise
        trace.phase(step,'sync',time.monotonic()-started,
            {'transfer_call_returned':True,'fixed_token_weight_probe_passed':False})
        if self.trainer.pilot_settings.get('scope')=='code_rl_v3' or self.trainer.pilot_settings.get('stop_generated_tokens'):
            import copy
            self.trainer._last_complete_data_state=copy.deepcopy(self.trainer.train_dataloader.state_dict())
            self.trainer._last_synced_step=step
        return result


def instrumented_trainer(base_class):
    """Construct lazily so tests never import Ray, Torch or start GPU workers."""
    class AuditedPilotTrainer(base_class):
        def __init__(self,*args,pilot_trace,pilot_settings,**kwargs):
            self.pilot_trace=pilot_trace;self.pilot_settings=pilot_settings
            self._pilot_generated={};self._old_audited_step=None
            super().__init__(*args,**kwargs)

        def init_workers(self):
            super().init_workers()
            self.async_rollout_manager=_ObservedRollout(self.async_rollout_manager,self)
            self.checkpoint_manager=_ObservedCheckpointManager(self.checkpoint_manager,self)

        def _checked_responses(self,batch):
            masks=plain(batch.batch['response_mask']);tokens=plain(batch.batch['responses'])
            ids=plain(batch.non_tensor_batch['request_id'])
            if len(set(ids))!=len(ids) or set(ids)!=set(self._pilot_generated):
                raise ValueError('generated_candidate_identity_changed_before_update')
            rows=[selected(t,m) for t,m in zip(tokens,masks,strict=True)]
            if any(self._pilot_generated[rid]!=row for rid,row in zip(ids,rows,strict=True)):
                raise ValueError('generated_tokens_changed_before_update')
            if sequence_identities(batch)!=self._pilot_sequences:
                raise ValueError('context_or_policy_mask_changed_before_update')
            return ids,rows,masks

        def _compute_old_log_prob(self,batch):
            ids,tokens,masks=self._checked_responses(batch)
            started=time.monotonic()
            try:
                output,mfu=super()._compute_old_log_prob(batch)
                elapsed=time.monotonic()-started
                values=plain(output.batch['old_log_probs'])
                rows=[{'request_id':rid,'tokens':t,'old_log_probs':selected(v,m)}
                    for rid,t,v,m in zip(ids,tokens,values,masks,strict=True)]
                path=self.pilot_trace.log_probs(self.global_steps,rows)
                self.pilot_trace.phase(self.global_steps,'old_log_prob',elapsed,
                    {'source':'actor_recomputed_before_update','artifact_sha256':file_digest(path)})
                self._old_audited_step=self.global_steps
                return output,mfu
            except BaseException as exc:
                self.pilot_trace.phase(self.global_steps,'old_log_prob',time.monotonic()-started,
                    {'failure_type':type(exc).__name__},complete=False)
                raise

        def _update_actor(self,batch):
            if self._old_audited_step!=self.global_steps:
                raise ValueError('actor_old_log_probs_must_be_recorded_before_update')
            ids,tokens,masks=self._checked_responses(batch)
            payload={'active_response_tokens':sum(map(len,tokens)),'active_sequences':len(ids),
                'loss_agg_mode':self.pilot_settings['loss_agg_mode'],
                'denominator_runtime_equivalence_verified':False}
            started=time.monotonic()
            try:
                output=super()._update_actor(batch)
                # Preserve finite scalar metrics; worker-side gradient/parameter
                # identity checks remain a separate required acceptance gate.
                metrics=plain(output.meta_info.get('metrics',{}))
                payload['metrics']={k:v for k,v in metrics.items() if type(v) in (int,float)}
                for value in payload['metrics'].values():finite(value)
                self.pilot_trace.phase(self.global_steps,'actor_update',time.monotonic()-started,payload)
                self._old_audited_step=None
                return output
            except BaseException as exc:
                self.pilot_trace.phase(self.global_steps,'actor_update',time.monotonic()-started,
                    {**payload,'failure_type':type(exc).__name__},complete=False)
                raise

        def _save_checkpoint(self):
            started=time.monotonic();super()._save_checkpoint()
            elapsed=time.monotonic()-started
            path=Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
            receipt=checkpoint_receipt(path)
            destination=self.pilot_trace.root/'checkpoints'/f'{self.pilot_trace.session}-{self.global_steps}.json'
            write_json(destination,{'checkpoint':str(path.resolve()),'receipt':receipt})
            self.pilot_trace.phase(self.global_steps,'checkpoint_save',elapsed,
                {'receipt_path':str(destination),'receipt_sha256':file_digest(destination),
                 'hashing_wall_s':time.monotonic()-started-elapsed})

        def _load_checkpoint(self):
            mode=self.config.trainer.resume_mode
            if mode not in ('disable','resume_path'):
                raise ValueError('explicit_resume_path_required_no_automatic_fallback')
            if mode=='resume_path':
                receipt=json.loads(Path(self.pilot_settings['resume_receipt']).read_text(encoding='utf-8'))
                checkpoint=Path(self.config.trainer.resume_from_path).resolve()
                if checkpoint!=Path(receipt['checkpoint']).resolve():raise ValueError('resume_receipt_path_mismatch')
                verify_checkpoint(checkpoint,receipt['receipt'])
            started=time.monotonic();result=super()._load_checkpoint()
            self.pilot_trace.phase(self.global_steps,'load_returned',time.monotonic()-started,
                {'mode':mode,'optimizer_rng_restore_equivalence_verified':False})
            if mode=='disable':self._save_checkpoint()  # authenticated pre-update comparison state
            return result

        def fit(self):
            try:return super().fit()
            except BudgetExceeded as exc:
                target_reached=isinstance(exc,GeneratedTokenTargetReached)
                if self.pilot_settings.get('scope')!='code_rl_v3' and not target_reached:raise
                # Reservation fails before generation/update at current global_steps.
                # Save exactly the previous completed update, not a partially run step.
                next_step=self.global_steps
                if next_step<=1 or getattr(self,'_last_synced_step',None)!=next_step-1:raise
                # The next batch was already fetched by the upstream loop. Restore
                # the data cursor captured at the last completed update/sync.
                self.train_dataloader.load_state_dict(self._last_complete_data_state)
                self.global_steps=next_step-1
                # The upstream loop synced weights (waking vLLM) before this reservation
                # failed. Free rollout memory before the final save, as upstream does
                # before loading a checkpoint; still attempt the save if sleeping fails.
                sleep=getattr(getattr(self,'checkpoint_manager',None),'sleep_replicas',None)
                if sleep is not None:
                    try:sleep()
                    except Exception as exc:
                        self.pilot_trace.phase(self.global_steps,'final_save_sleep_failed',0.,
                            {'failure_type':type(exc).__name__},complete=False)
                local=Path(self.config.trainer.default_local_dir)
                latest=local/'latest_checkpointed_iteration.txt'
                # A save_freq checkpoint of this very step needs no second save.
                if not (latest.is_file() and latest.read_text().strip()==str(self.global_steps)):
                    self._save_checkpoint()
                ledger=self.pilot_trace.ledger.snapshot()
                completion={'completion_reason':'generated_token_target_reached' if target_reached else 'normal_token_budget_end',
                    'last_complete_update':self.global_steps,
                    'checkpoint':str(local/f'global_step_{self.global_steps}'),
                    'unused_tokens':ledger['remaining']}
                if target_reached:
                    completion.update(stop_generated_tokens=self.pilot_settings['stop_generated_tokens'],
                        lineage_generated_tokens=lineage_generated_tokens(ledger,next_step))
                write_json(self.pilot_trace.root/'completion.json',completion)
                self.global_steps=next_step
            finally:write_json(self.pilot_trace.root/('observations-'+self.pilot_trace.session+'.json'),
                self.pilot_trace.snapshot())

    return AuditedPilotTrainer
