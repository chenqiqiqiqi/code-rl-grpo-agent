import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('summary',Path(__file__).resolve().parents[1]/'scripts/summarize_stage1_pool.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_native_summary_never_claims_zero_cloud_cost_or_gpu_throughput():
    context=module.execution_context('native_linux')
    assert context['external_paid_cny'] is None
    assert context['native_CPU_validated'] is True
    assert context['timing_limitation']=='native_CPU_reference_backtest_not_GPU_pipeline_throughput'


def test_qemu_and_unknown_context():
    assert module.execution_context('qemu_tcg')['native_CPU_validated'] is False
    with pytest.raises(ValueError):module.execution_context('gpu')
