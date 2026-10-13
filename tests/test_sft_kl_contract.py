import json
from pathlib import Path
import pytest

def test_kl_objective_requires_explicit_finite_weight():
    from posttrain.training.sft_kl_objective import checked_objective
    cfg=json.loads(Path('configs/sft_selfdistill_20260929.json').read_text())
    good={**cfg,'objective':'ce_frozen_m0_forward_kl_v1','kl_beta':1.0}
    assert checked_objective(good)['kl_beta']==1.0
    for extra in ({},{'objective':'ce'},{'objective':'ce_frozen_m0_forward_kl_v1','kl_beta':-1},
                  {'objective':'ce_frozen_m0_forward_kl_v1','kl_beta':float('nan')},
                  {'objective':'ce_frozen_m0_forward_kl_v1','kl_beta':True}):
        with pytest.raises(ValueError):checked_objective({**cfg,**extra})
