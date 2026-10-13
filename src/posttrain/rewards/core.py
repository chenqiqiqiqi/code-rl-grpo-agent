from __future__ import annotations

import statistics

from posttrain.execution.contracts import JudgeResult


def rewards(result: JudgeResult, *, binary_only=False) -> dict[str,float] | None:
    if binary_only:
        return {'binary':float(result.passed == result.total)} if result.binary_valid else None
    if not result.valid:
        return None
    binary = float(result.passed == result.total)
    return {"binary":binary,"mixed":.5*result.passed/result.total+.5*binary}


def group_statistics(groups: list[list[JudgeResult]], expected_g: int) -> dict:
    if expected_g < 2:
        raise ValueError("G must be >= 2")
    valid,values,mask = 0,{"binary":[],"mixed":[]},[]
    for group in groups:
        ok = len(group)==expected_g and all(x.valid for x in group)
        mask.append(ok)
        if ok:
            valid += 1
            for kind in values:
                values[kind].append([rewards(x)[kind] for x in group])
    out = {"generated_groups":len(groups),"valid_groups":valid,
           "valid_group_fraction":valid/len(groups) if groups else None,"valid_group_mask":mask}
    for kind,rows in values.items():
        signal = sum(max(row)-min(row)>1e-8 for row in rows)
        out[kind] = {"signal_groups":signal,"signal_group_fraction":signal/valid if valid else None,
                     "all_zero_groups":sum(all(v==0 for v in row) for row in rows),
                     "all_one_groups":sum(all(v==1 for v in row) for row in rows),
                     "zero_variance_groups":sum(max(row)-min(row)<=1e-8 for row in rows),
                     "group_population_stdev":[statistics.pstdev(row) for row in rows]}
    # Numerical diagnostics alone are not evidence of a real 200-problem calibration.
    out['diagnostic_thresholds_met'] = bool(valid and out['valid_group_fraction']>=.95 and out['binary']['signal_group_fraction']>=.10)
    return out
