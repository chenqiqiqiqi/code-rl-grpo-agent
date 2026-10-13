import pytest

from scripts.probe_sft_batching import assess_batching_equivalence


def probe_rows(*, configured_gradient=0.014, configured_loss=0.001):
    # BF16 can change the gradient direction when singleton and batched kernels
    # take different numerical paths. The production microbatch is 2, so the
    # acceptance check must use microbatch 2 against the full batch of 4.
    return [
        {
            'micro_batch_size': 1,
            'gradient_relative_l2_vs_full_batch': 0.15,
            'loss_absolute_difference_vs_full_batch': 0.001,
        },
        {
            'micro_batch_size': 2,
            'gradient_relative_l2_vs_full_batch': configured_gradient,
            'loss_absolute_difference_vs_full_batch': configured_loss,
        },
        {
            'micro_batch_size': 4,
            'gradient_relative_l2_vs_full_batch': 0.0,
            'loss_absolute_difference_vs_full_batch': 0.0,
        },
    ]


def test_acceptance_uses_configured_microbatch_against_full_batch():
    result = assess_batching_equivalence(
        probe_rows(), batch_size=4, micro_batch_size=2
    )

    assert result['passed'] is True
    assert result['reference_micro_batch_size'] == 4
    assert result['tested_micro_batch_size'] == 2
    assert result['gradient_relative_l2'] == pytest.approx(0.014)
    assert result['loss_absolute_difference'] == pytest.approx(0.001)
    assert result['diagnostic_micro_batch_sizes'] == [1, 2, 4]
    assert result['tolerance'] == {
        'gradient_relative_l2': 0.05,
        'loss_absolute_difference': 0.02,
    }


def test_acceptance_fails_when_configured_gradient_drift_exceeds_tolerance():
    result = assess_batching_equivalence(
        probe_rows(configured_gradient=0.051), batch_size=4, micro_batch_size=2
    )

    assert result['passed'] is False
    assert result['failure_reasons'] == ['gradient_relative_l2_exceeds_tolerance']


def test_acceptance_fails_when_configured_loss_drift_exceeds_tolerance():
    result = assess_batching_equivalence(
        probe_rows(configured_loss=0.021), batch_size=4, micro_batch_size=2
    )

    assert result['passed'] is False
    assert result['failure_reasons'] == ['loss_difference_exceeds_tolerance']


def test_acceptance_requires_both_configured_and_full_batch_rows():
    rows = [row for row in probe_rows() if row['micro_batch_size'] != 4]
    with pytest.raises(ValueError, match='missing_full_batch_reference'):
        assess_batching_equivalence(rows, batch_size=4, micro_batch_size=2)

    rows = [row for row in probe_rows() if row['micro_batch_size'] != 2]
    with pytest.raises(ValueError, match='missing_configured_micro_batch'):
        assess_batching_equivalence(rows, batch_size=4, micro_batch_size=2)
