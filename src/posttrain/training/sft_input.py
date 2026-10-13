"""GPU-side SFT examples contain supervised answers, never hidden test fixtures."""
from dataclasses import dataclass

from posttrain.data.schema import Problem


@dataclass(frozen=True)
class SFTExample:
    problem_id: str
    cluster_id: str
    messages: list[dict]
    references: tuple[str, ...]
    validation_status: str = 'reference_passed'

    def prompt(self):
        return self.messages


def load_example(row):
    if row.get('schema') != 'public_sft_v1':
        return Problem.from_dict(row)  # compatibility with existing CPU-side workflows
    if set(row) != {'schema','problem_id','cluster_id','messages','answer','source_row_sha256'}:
        raise ValueError('unexpected_sft_fields')
    if not row['problem_id'] or not row['cluster_id'] or not isinstance(row['answer'],str) or not row['answer'].strip():
        raise ValueError('invalid_sft_identity_or_answer')
    if len(row['messages']) != 2 or [m['role'] for m in row['messages']] != ['system','user']:
        raise ValueError('invalid_sft_messages')
    if any(set(m) != {'role','content'} or not isinstance(m['content'],str) for m in row['messages']):
        raise ValueError('invalid_sft_message_fields')
    return SFTExample(row['problem_id'],row['cluster_id'],row['messages'],(row['answer'],))
