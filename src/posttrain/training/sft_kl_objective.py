"""Explicit full-vocabulary forward KL on supervised assistant prefixes."""
import math

OBJECTIVE='ce_frozen_m0_forward_kl_v1'

def checked_objective(cfg):
    if cfg.get('objective')!=OBJECTIVE:
        raise ValueError('explicit_supported_kl_objective_required')
    beta=cfg.get('kl_beta')
    if isinstance(beta,bool) or not isinstance(beta,(int,float)) or not math.isfinite(beta) or beta<0:
        raise ValueError('finite_nonnegative_kl_beta_required')
    return cfg

def teacher_log_probs(logits,labels):
    import torch
    with torch.no_grad():
        return torch.log_softmax(logits[:,:-1][labels[:,1:]!=-100].float(),dim=-1).detach()

def masked_forward_kl(logits,labels,teacher_lp,chunk_size=32):
    import torch
    selected=logits[:,:-1][labels[:,1:]!=-100]
    if selected.shape!=teacher_lp.shape or selected.shape[0]==0:
        raise ValueError('teacher_student_shifted_mask_mismatch')
    if teacher_lp.requires_grad:
        raise ValueError('teacher_must_be_detached')
    total=logits.new_zeros((),dtype=torch.float32)
    for start in range(0,len(selected),chunk_size):
        t=teacher_lp[start:start+chunk_size]
        s=torch.log_softmax(selected[start:start+chunk_size].float(),dim=-1)
        total=total+(t.exp()*(t-s)).sum()
    return total/len(selected)
