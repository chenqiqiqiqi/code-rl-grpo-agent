from __future__ import annotations

import math
from pathlib import Path

from posttrain.common import digest,file_digest


def require_real_pilot(report: dict) -> None:
    if report.get('evidence_level')!='real_7b_pilot' or report.get('model_id')!='Qwen/Qwen2.5-Coder-7B-Instruct':
        raise ValueError('real_qwen_7b_pilot_required')
    checks=('lora_changed','base_unchanged','rollout_sync_fixed_token_probe',
            'old_log_prob_source_audited','loss_denominator_audited','optimizer_restored',
            'scheduler_restored','rng_restored','data_progress_restored','ledger_not_rolled_back')
    if any(report.get('checks',{}).get(k) is not True for k in checks):
        raise ValueError('incomplete_pilot_acceptance')
    if report.get('stable_complete_steps',0)<20:
        raise ValueError('20_stable_complete_pilot_steps_required')
    for key in ('generation','queue','judging','log_prob','update','sync'):
        times=report.get('timing_s',{}).get(key,[])
        if len(times)<20 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in times):
            raise ValueError('missing_measured_phase_timing:'+key)


def validate_freeze(record: dict, root: Path) -> None:
    """Validation only. This does not generate imaginary proof or a freeze record."""
    required=('common_start','reference','model_revision','training_manifest','development_manifest',
              'calibration','final_manifest','software_lock','generation','optimization',
              'log_prob','checkpoint_rule','budget','pilot_evidence','dedup_evidence')
    if any(not record.get(k) for k in required):
        raise ValueError('incomplete_protocol_freeze')
    if record.get('seeds')!=[42,43] or record.get('rewards')!={'binary':'all_pass','mixed':[0.5,0.5]}:
        raise ValueError('protocol_matrix_mismatch')
    require_real_pilot(record['pilot_evidence'])
    calibration=record['calibration']
    if (calibration.get('evidence_level')!='isolated_execution' or calibration.get('problems')!=200
        or calibration.get('common_start')!=record['common_start']
        or calibration.get('valid_group_fraction',0)<.95
        or calibration.get('binary_signal_group_fraction',0)<.10):
        raise ValueError('final_common_start_calibration_not_passed')
    if record['checkpoint_rule']!='last_complete_update_at_normal_token_budget_end':
        raise ValueError('invalid_primary_checkpoint_rule')
    if record['budget']['forecast_total_cny']>1600 or record['budget']['emergency_unallocated_cny']<400:
        raise ValueError('budget_not_ready')
    for artifact in record.get('artifacts',[]):
        if file_digest(root/artifact['path'])!=artifact['sha256']:
            raise ValueError('artifact_changed_since_freeze')
