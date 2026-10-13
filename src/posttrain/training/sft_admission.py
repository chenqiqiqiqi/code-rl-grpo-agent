"""Fail-closed gold admission; heldout input contains clusters only, never answers."""
import re
from posttrain.common import digest
from posttrain.training.sft_input import load_example

def checked_admitted_rows(rows,manifest,heldout):
    if not isinstance(manifest,dict) or manifest.get('schema')!='sft_gold_admission_v1':raise ValueError('admission_manifest_required')
    if set(heldout)!={'schema','clusters'} or heldout['schema']!='heldout_clusters_v1':raise ValueError('heldout_clusters_only_required')
    if not heldout['clusters'] or manifest.get('heldout_sha256')!=digest(heldout):raise ValueError('heldout_identity_changed')
    entries=manifest.get('entries',[])
    if not rows or len(entries)!=len(rows):raise ValueError('admission_row_count')
    if len({r['problem_id'] for r in rows})!=len(rows):raise ValueError('duplicate_admission_id')
    for r,e in zip(rows,entries):
        if (e.get('role')!='S_gold' or e.get('split')!='train' or e.get('whole_pass') is not True
            or e.get('row_sha256')!=digest(r) or e.get('problem_id')!=r.get('problem_id')
            or e.get('cluster_id')!=r.get('cluster_id')):raise ValueError('unadmitted_or_wrong_purpose')
        if not re.fullmatch('[0-9a-f]{64}',e.get('receipt_sha256','')) or not e.get('executor_version'):raise ValueError('execution_receipt_required')
        limits=e.get('limits',{})
        if limits.get('memory_mb')!=256 or not 0<limits.get('time_s',0)<=30:raise ValueError('formal_resource_limits_required')
    if {r['cluster_id'] for r in rows}&set(heldout['clusters']):raise ValueError('heldout_cluster_leak')
    return [load_example(r) for r in rows]
