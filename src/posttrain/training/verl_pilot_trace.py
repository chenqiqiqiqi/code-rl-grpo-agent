"""Durable pilot observations; never manufactures training acceptance evidence.

This module is GPU-independent so failure and resume accounting can be tested on
Windows. The ledger lives beside, not inside, replaceable model checkpoints.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import sqlite3
import uuid

from posttrain.common import canonical,file_digest,utc_now,write_json
from posttrain.training.budget import TokenLedger


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('nonfinite_or_missing_observation')
    return value


def checkpoint_receipt(root):
    """Authenticate complete single-GPU FSDP state without deserializing pickle.

File presence/hashes prove transport integrity, not successful optimizer/RNG
restoration. The latter must be checked on the real training worker.
"""
    root=Path(root)
    required={'data_progress':'data.pt','model':'actor/model_world_size_1_rank_0.pt',
        'optimizer':'actor/optim_world_size_1_rank_0.pt',
        'scheduler_and_rng':'actor/extra_state_world_size_1_rank_0.pt'}
    for name,path in required.items():
        if not (root/path).is_file():raise ValueError('checkpoint_missing_'+name)
    paths=sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('checkpoint_symlink_rejected')
    return {'files':{p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':file_digest(p)}
        for p in paths},'required_state':required,'restore_equivalence_verified':False}


def verify_checkpoint(root,receipt):
    if checkpoint_receipt(root)!=receipt:raise ValueError('checkpoint_changed_since_receipt')


class PilotTrace:
    def __init__(self,root,identity,cap,resume=False):
        self.root=Path(root)
        if (identity.get('scope') not in ('engineering_pilot_only','code_rl_v3')
            or not re.fullmatch('[a-f0-9]{32}',identity.get('run_uuid',''))
            or not re.fullmatch('[a-f0-9]{64}',identity.get('config_sha256',''))):
            raise ValueError('explicit_pilot_identity_required')
        identity_path=self.root/'identity.json'
        if resume:
            if not identity_path.is_file() or json.loads(identity_path.read_text(encoding='utf-8'))!=identity:
                raise ValueError('pilot_resume_identity_changed')
            if not (self.root/'tokens.sqlite').is_file() or not (self.root/'events.sqlite').is_file():
                raise ValueError('pilot_resume_ledger_missing')
        else:
            self.root.mkdir(parents=True,exist_ok=False)
            write_json(identity_path,identity)
        self.session=uuid.uuid4().hex
        self.ledger=TokenLedger(self.root/'tokens.sqlite',cap)
        self.db=sqlite3.connect(self.root/'events.sqlite',isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS events '
            '(id INTEGER PRIMARY KEY,session TEXT,step INTEGER,phase TEXT,complete INTEGER,elapsed_s REAL,payload TEXT,created_at TEXT)')
        self.phase(0,'session_start',0.,{'resume':resume,'charged_tokens':self.ledger.used()})

    def __enter__(self):return self

    def __exit__(self,*args):self.close()

    def close(self):
        self.ledger.close();self.db.close()

    def phase(self,step,name,elapsed_s,payload=None,complete=True):
        if type(step) is not int or step<0 or finite(elapsed_s)<0:
            raise ValueError('invalid_phase_observation')
        self.db.execute('INSERT INTO events(session,step,phase,complete,elapsed_s,payload,created_at) '
            'VALUES (?,?,?,?,?,?,?)',(self.session,step,name,int(complete),elapsed_s,canonical(payload or {}),utc_now()))

    def begin_rollout(self,step,prompts,g,max_tokens):
        rid=self.ledger.reserve(prompts,g,max_tokens,f'pilot:{self.session}:{step}')
        self.phase(step,'rollout_started',0.,{'reservation_id':rid},complete=False)
        return rid

    def finish_rollout(self,rid,candidates,timing,wall_s):
        reservations={r['id']:r for r in self.ledger.snapshot()['reservations']}
        if rid not in reservations or reservations[rid]['actual'] is not None:
            raise ValueError('unknown_or_completed_rollout')
        reservation=reservations[rid];plan=reservation['payload']
        if not plan['purpose'].startswith(f'pilot:{self.session}:'):
            raise ValueError('cannot_complete_prior_session_unknown_rollout')
        step=int(plan['purpose'].rsplit(':',1)[1])
        if len(candidates)!=plan['prompts']*plan['g']:
            raise ValueError('incomplete_candidate_count')
        groups=Counter(c['uid'] for c in candidates)
        if len(groups)!=plan['prompts'] or set(groups.values())!={plan['g']}:
            raise ValueError('incomplete_candidate_group')
        if len({c['request_id'] for c in candidates})!=len(candidates):
            raise ValueError('duplicate_scoring_request')
        for row in candidates:
            response=row['response_tokens']
            if not 0<len(response)<=plan['max_tokens']:raise ValueError('response_token_limit')
            if len(response)!=len(row['rollout_log_probs']):raise ValueError('rollout_log_prob_length')
            if any(type(v) is not int or v<0 for v in response+row['prompt_tokens']):
                raise ValueError('invalid_token_ids')
            for value in row['rollout_log_probs']:finite(value)
            fast = row.get('scoring_mode','full') == 'binary_fail_fast'
            mixed_ok = (row.get('mixed_available') is False and row['mixed'] is None) if fast else (
                0<=finite(row['mixed'])<=1)
            if row['binary'] not in (0.,1.) or not mixed_ok:
                raise ValueError('invalid_pilot_reward')
            if finite(row['queue_s'])<0 or finite(row['judging_s'])<0:
                raise ValueError('invalid_scoring_timing')
        for value in timing.values():finite(value)
        if finite(wall_s)<0:raise ValueError('invalid_rollout_timing')
        artifact=self.root/'rollouts'/(rid+'.json')
        # Persist exact generated tokens before reducing the worst-case charge.
        write_json(artifact,{'session':self.session,'step':step,'reservation_id':rid,
            'candidates':candidates,'timing_s':timing,'generation_and_scoring_wall_s':wall_s})
        actual=sum(len(c['response_tokens']) for c in candidates)
        self.ledger.settle(rid,actual)
        self.phase(step,'rollout',wall_s,{'reservation_id':rid,'artifact_sha256':file_digest(artifact),
            'generated_tokens':actual,'candidate_count':len(candidates),
            'input_tokens':sum(len(c['prompt_tokens']) for c in candidates)})

    def log_probs(self,step,rows):
        if not rows:raise ValueError('missing_old_log_prob_rows')
        for row in rows:
            if len(row['tokens'])!=len(row['old_log_probs']):raise ValueError('old_log_prob_length')
            for value in row['old_log_probs']:finite(value)
        path=self.root/'old_log_probs'/f'{self.session}-{step}.json'
        if path.exists():raise ValueError('old_log_prob_observation_already_exists')
        write_json(path,{'source':'actor_recomputed_before_update','rows':rows,'step':step,'session':self.session})
        return path

    def snapshot(self):
        rows=self.db.execute('SELECT session,step,phase,complete,elapsed_s,payload,created_at '
            'FROM events ORDER BY id').fetchall()
        events=[{'session':s,'step':step,'phase':p,'complete':bool(c),'elapsed_s':t,
            'payload':json.loads(data),'created_at':created} for s,step,p,c,t,data,created in rows]
        phases={}
        for event in events:
            if event['complete']:
                phases.setdefault((event['session'],event['step']),set()).add(event['phase'])
        complete=[{'session':s,'step':step} for (s,step),names in phases.items()
            if {'rollout','old_log_prob','actor_update','sync'}<=names]
        return {'scope':'engineering_pilot_observations','real_7b_pilot_complete':False,
            'tokens':self.ledger.snapshot(),'events':events,
            'completed_updates':sum(e['phase']=='actor_update' and e['complete'] for e in events),
            'complete_update_and_sync_steps':complete}
