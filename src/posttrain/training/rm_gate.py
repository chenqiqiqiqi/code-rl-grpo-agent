"""Preregistered contemporaneous zero-adapter control stop rule."""
def control_gate(m0,z):
    def index(rows):
        if len(rows)!=348 or len({r['problem_id'] for r in rows})!=348:raise ValueError('full348_required')
        if any(r['state']!='complete' or type(r['binary_success']) is not bool for r in rows):raise ValueError('valid_scores_required')
        return {r['problem_id']:r['binary_success'] for r in rows}
    a,b=index(m0),index(z)
    if a.keys()!=b.keys():raise ValueError('control_problem_mismatch')
    gain=sum(b[k] and not a[k] for k in a);loss=sum(a[k] and not b[k] for k in a)
    return {'gain':gain,'loss':loss,'flips':gain+loss,'net':gain-loss,'M0_pass':sum(a.values()),'Z_pass':sum(b.values()),'passed':gain+loss<=20 and abs(gain-loss)<5}
