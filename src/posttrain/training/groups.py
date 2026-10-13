from __future__ import annotations

from dataclasses import dataclass

from posttrain.execution.contracts import JudgeResult
from posttrain.rewards import rewards
from posttrain.training.budget import TokenLedger


class IncompleteBatch(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateGroup:
    problem_id: str
    generated_tokens: tuple[int,...]
    results: tuple[JudgeResult,...]

    def valid(self,g: int) -> bool:
        return (len(self.results)==len(self.generated_tokens)==g and all(r.valid for r in self.results)
                and len({r.tests_hash for r in self.results})==1
                and len({(r.executor_version,r.evidence_level) for r in self.results})==1)


def gather_valid_batch(primary: list[str], replacement_pool: list[str], g: int, max_tokens: int,
                       ledger: TokenLedger, generate_and_score, reward_kind: str) -> dict:
    """Complete-group replacement, at most one batch's worth, with durable token debit.

    generate_and_score(ids) must return CandidateGroups and exact token counts. An
    exception leaves the reservation charged at its worst case until reconciled.
    This is a tested integration primitive; not yet wired into the pinned verl loop.
    """
    if reward_kind not in ('binary','mixed') or g<2 or not primary:
        raise ValueError('invalid_batch_configuration')
    target=len(primary)
    accepted,attempts=[],[]
    def run(ids,purpose):
        reservation=ledger.reserve(len(ids),g,max_tokens,purpose)
        groups=generate_and_score(ids)
        if len(groups)!=len(ids) or [r.problem_id for r in groups]!=ids:
            raise ValueError('generator_changed_registered_prompt_order')
        if any(len(x.generated_tokens)!=g or any(type(t) is not int or not 0<=t<=max_tokens for t in x.generated_tokens) for x in groups):
            raise ValueError('missing_or_invalid_generation_accounting')
        ledger.settle(reservation,sum(sum(x.generated_tokens) for x in groups))
        for group in groups:
            valid=group.valid(g)
            attempts.append({'problem_id':group.problem_id,'valid':valid,'purpose':purpose,
                             'tokens':sum(group.generated_tokens),'reservation':reservation})
            if valid:
                accepted.append(group)
    run(primary,'primary')
    remaining=target-len(accepted)
    if remaining:
        # Consume a predetermined prefix, never search by observed reward values.
        replacements=replacement_pool[:remaining]
        if len(replacements)!=remaining:
            raise IncompleteBatch('replacement_pool_exhausted_no_partial_update')
        run(replacements,'replacement')
    if len(accepted)!=target:
        raise IncompleteBatch('one_replacement_round_exhausted_no_partial_update')
    return {'groups':accepted,'rewards':[[rewards(r)[reward_kind] for r in group.results] for group in accepted],
            'attempts':attempts,'replacement_prompts_consumed':remaining}
