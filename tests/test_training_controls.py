import pytest

from posttrain.training.budget import BudgetExceeded, TokenLedger
from posttrain.training.guards import SignalMonitor, main_checkpoint, select_sft_start


def test_token_reservation_is_durable_across_resume(tmp_path):
    path=tmp_path/'ledger.sqlite'
    ledger=TokenLedger(path,100)
    reservation=ledger.reserve(2,4,10,'rollout')
    assert ledger.used()==80
    with pytest.raises(BudgetExceeded):
        ledger.reserve(1,4,10,'replacement')
    ledger.close()
    resumed=TokenLedger(path,100)
    assert resumed.used()==80
    resumed.settle(reservation,32)
    resumed.settle(reservation,32)
    with pytest.raises(ValueError):
        resumed.settle(reservation,0)
    with pytest.raises(ValueError):
        TokenLedger(path,101)
    assert resumed.used()==32
    resumed.close()


def test_unknown_generation_reservation_is_not_automatically_refunded(tmp_path):
    path=tmp_path/'ledger.sqlite'
    first=TokenLedger(path,80)
    first.reserve(2,4,10,'timed_out_generation')
    second=TokenLedger(path,80)
    with pytest.raises(BudgetExceeded):
        second.reserve(1,1,1,'new_generation')
    first.close();second.close()


def test_stop_windows_and_invalid_groups():
    monitor=SignalMonitor()
    assert monitor.observe(16,0,0)=='no_valid_groups'
    monitor=SignalMonitor()
    assert all(monitor.observe(100,100,1) is None for _ in range(39))
    assert monitor.observe(100,100,1)=='low_signal_two_windows'
    monitor=SignalMonitor()
    assert all(monitor.observe(16,16,0) is None for _ in range(19))
    assert monitor.observe(16,16,0)=='zero_signal_20_steps'
    monitor=SignalMonitor()
    assert all(monitor.observe(16,15,1) is None for _ in range(9))
    assert monitor.observe(16,15,1)=='valid_group_fraction_below_95_percent_for_10_steps'


def test_sft_ties_fallback_and_last_complete_update():
    cs=[{'step':10,'dev_greedy_pass1':.3,'complete':True},{'step':5,'dev_greedy_pass1':.3,'complete':True}]
    assert select_sft_start(.3,cs,False)['checkpoint']['step']==5
    assert select_sft_start(.4,cs,False)['action']=='review_and_one_correction'
    assert select_sft_start(.4,cs,True)['action']=='fallback_m0'
    with pytest.raises(ValueError):
        main_checkpoint({'completion_reason':'low_signal','last_complete_update':'earlier'})
    assert main_checkpoint({'completion_reason':'normal_token_budget_end','last_complete_update':'last'})=='last'
