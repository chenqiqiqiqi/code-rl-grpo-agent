"""Read-only diagnostic input validation; raw trajectories are not CE admission."""
def validate_sample(s):
    x,y,a=s['input_ids'],s['labels'],s['attention_mask']
    if not 1<len(x)<=4096 or len(x)!=len(y) or len(x)!=len(a):raise ValueError('shape')
    if any(v!=1 for v in a) or y[0]!=-100:raise ValueError('mask')
    target=[i for i,v in enumerate(y) if v!=-100]
    if not target or target!=list(range(target[0],len(x))):raise ValueError('target_span')
    if any(y[i]!=x[i] for i in target):raise ValueError('label_identity')
    return len(target)

def trajectory_sample(prefix,ids,input_tokens,output_tokens,eos):
    if len(prefix)!=input_tokens or len(ids)!=output_tokens or not ids or ids[-1]!=eos:
        raise ValueError('raw_trajectory_identity')
    s={'input_ids':prefix+ids,'labels':[-100]*len(prefix)+ids,'attention_mask':[1]*(len(prefix)+len(ids))}
    validate_sample(s)
    return s

def diagnostic_losses(selected_logits,targets,teacher_lp):
    import torch
    if selected_logits.shape!=teacher_lp.shape or len(targets)!=len(teacher_lp) or teacher_lp.requires_grad:
        raise ValueError('teacher_mask_or_detach')
    lp=torch.log_softmax(selected_logits,dim=-1)
    ce=-lp.gather(1,targets[:,None]).mean()
    kl=(teacher_lp.exp()*(teacher_lp-lp)).sum()/len(targets)
    return ce,kl

def gradient_metrics(ce,kl,beta=1.0):
    import torch
    if not torch.isfinite(ce).all() or not torch.isfinite(kl).all():raise ValueError('nonfinite_gradient')
    # FP64 reductions without keeping extra whole-vector FP64 copies.
    cc=kk=ck=0.0
    for start in range(0,ce.numel(),1048576):
        c=ce[start:start+1048576].double();k=kl[start:start+1048576].double()*beta
        cc+=float(torch.dot(c,c));kk+=float(torch.dot(k,k));ck+=float(torch.dot(c,k))
    cn,kn=cc**.5,kk**.5
    return {'ce_gradient_norm':cn,'beta_kl_gradient_norm':kn,
        'kl_to_ce_gradient_norm':kn/cn if cn else None,'cosine':ck/(cn*kn) if cn and kn else None}
