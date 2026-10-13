from __future__ import annotations


class SignalMonitor:
    """Protocol section 8; 20-step windows are nonoverlapping."""
    def __init__(self):
        self.valid_low_steps = 0
        self.window = []
        self.previous_low = False

    def observe(self, generated: int, valid: int, signal: int) -> str | None:
        if not 0 <= signal <= valid <= generated or generated == 0:
            raise ValueError('invalid_group_counts')
        if valid == 0:
            return 'no_valid_groups'
        self.valid_low_steps = self.valid_low_steps+1 if valid/generated < .95 else 0
        if self.valid_low_steps >= 10:
            return 'valid_group_fraction_below_95_percent_for_10_steps'
        self.window.append((valid,signal))
        if len(self.window) == 20:
            v,s = map(sum,zip(*self.window))
            self.window.clear()
            low = s/v < .10
            stop = 'zero_signal_20_steps' if s==0 else 'low_signal_two_windows' if low and self.previous_low else None
            self.previous_low = low
            return stop
        return None


def select_sft_start(m0_score: float, candidates: list[dict], correction_used: bool) -> dict:
    if not candidates or any(not c.get('complete') for c in candidates):
        raise ValueError('only_preregistered_complete_sft_checkpoints_allowed')
    chosen = min(candidates,key=lambda c:(-c['dev_greedy_pass1'],c['step']))
    if chosen['dev_greedy_pass1'] < m0_score-.02:
        return {'action':'fallback_m0' if correction_used else 'review_and_one_correction', 'retained_m1':chosen}
    return {'action':'use_sft','checkpoint':chosen}


def main_checkpoint(run: dict) -> str:
    if run.get('completion_reason') != 'normal_token_budget_end' or not run.get('last_complete_update'):
        raise ValueError('incomplete_run_has_no_primary_checkpoint')
    return run['last_complete_update']
