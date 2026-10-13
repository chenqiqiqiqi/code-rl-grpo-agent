"""Independent supervised-gold and preservation-only data contracts."""
from posttrain.training.sft import update_plan
from posttrain.training.kl_diagnostic import validate_sample

def validate_routes(s,a,heldout_clusters):
    if not s or not a:raise ValueError('empty_route')
    for rows,role,admission in [(s,'supervised_gold','semantic_and_formal_pass'),(a,'preservation_only','frozen_m0_trajectory')]:
        for r in rows:
            if r.get('role')!=role or r.get('admission')!=admission:raise ValueError('supervision_or_preservation_purpose_mismatch')
            if not r.get('cluster_id') or not r.get('problem_id'):raise ValueError('missing_identity')
            validate_sample(r['sample'])
        if len({r['problem_id'] for r in rows})!=len(rows):raise ValueError('duplicate_route_id')
    sc={r['cluster_id'] for r in s};ac={r['cluster_id'] for r in a};hc=set(heldout_clusters)
    if sc&ac or sc&hc or ac&hc:raise ValueError('route_cluster_overlap')

def route_plan(s_size,a_size,batch_size,steps,seed):
    return update_plan(s_size,batch_size,steps,seed),update_plan(a_size,a_size,steps,seed+1)

def route_weights(s_counts,a_counts):
    def weights(xs):
        if not xs or any(type(x) is not int or x<=0 for x in xs):raise ValueError('invalid_route_token_count')
        return [x/sum(xs) for x in xs]
    return weights(s_counts),weights(a_counts)
