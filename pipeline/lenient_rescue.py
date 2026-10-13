"""Format-insensitive ("lenient") scoring, protocol v4 addendum 9: rescue samples that failed only on format.

Only samples that the strict pipeline rejected on format (single-turn) or that never produced a valid
submission (Agent: invalid action, tool/turn/generation limit) are re-extracted and re-scored; every other
sample keeps its strict judgment. Rescue extraction: the last ```python block (or unlabeled block) that parses
and defines a function, reduced to its top-level imports / function and class definitions / assignments
(self-test calls such as print(...) would otherwise write to stdout during the hidden tests).

  agent  (GPU host, needs the tokenizer): STATE PROMPTS OUT_CANDS
         decode the last generated turn of every non-submitted episode.
  single (local): CANDS STRICT_DIR OUT_CANDS          candidates whose strict record is format_rejected.
  merge  (local): STRICT_DIR RESCUE_CANDS RESCUE_DIR OUT_DIR
         OUT_DIR/records = strict records, with rescued samples replaced by their rescue records
         (marked lenient_rescued); compare_any.py reads it like any scoring directory.
Rescue request ids are sha256('lenient|' + original request id)[:32].
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def lenient_code(text):
    blocks = re.findall(r'```(?:python|py)?[ \t]*\r?\n(.*?)```', text, flags=re.S | re.I)
    for block in reversed(blocks):
        try:
            tree = ast.parse(block)
        except (SyntaxError, ValueError, RecursionError):
            continue
        body = [n for n in tree.body if isinstance(n, KEEP)]
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in body):
            return ast.unparse(ast.Module(body=body, type_ignores=[])) + '\n'
    return None


def rid(original):
    return hashlib.sha256(('lenient|' + original).encode()).hexdigest()[:32]


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8', newline='\n')


def agent(state, prompts, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/qwen7b', local_files_only=True)
    meta = {p['problem_id']: p for p in load(prompts)}
    rows, stats = [], {'episodes': 0, 'not_submitted': 0, 'rescued': 0}
    for st in load(state):
        stats['episodes'] += 1
        if st['episode']['end_reason'] == 'submitted' or not st['turn_ids']:
            continue
        stats['not_submitted'] += 1
        code = lenient_code(tok.decode(st['turn_ids'][-1], skip_special_tokens=True))
        if code is None:
            continue
        original = hashlib.sha256(st['eid'].encode()).hexdigest()[:32]  # = agent_eval.py finalize request_id
        p = meta[st['problem_id']]
        rows.append({'request_id': rid(original), 'orig_request_id': original, 'eid': st['eid'], 'problem_id': st['problem_id'],
                     'candidate_index': st['k_index'], 'text': '```python\n' + code + '```\n', 'finish_reason': 'stop',
                     'output_tokens': st['generated_count'], 'end_reason': st['episode']['end_reason'],
                     'tests_hash': p['tests_hash'], 'test_count': p['test_count']})
        stats['rescued'] += 1
    write(out, rows); print(json.dumps(stats))


def single(cands, strict_dir, out):
    strict = {json.loads(f.read_text())['request_id']: json.loads(f.read_text()) for f in Path(strict_dir, 'records').glob('*.json')}
    rows, stats = [], {'candidates': 0, 'format_rejected': 0, 'rescued': 0}
    for c in load(cands):
        stats['candidates'] += 1
        if not strict[c['request_id']]['format_rejected']:
            continue
        stats['format_rejected'] += 1
        code = lenient_code(c['text'])
        if code is None:
            continue
        rows.append({**c, 'request_id': rid(c['request_id']), 'orig_request_id': c['request_id'],
                     'text': '```python\n' + code + '```\n'})
        stats['rescued'] += 1
    write(out, rows); print(json.dumps(stats))


def merge(strict_dir, rescue_cands, rescue_dir, out_dir):
    out = Path(out_dir)/'records'
    if out.parent.exists():
        raise SystemExit('output exists')
    out.mkdir(parents=True)
    rescued = {c['orig_request_id']: c['request_id'] for c in load(rescue_cands)}
    n = {'strict': 0, 'rescued': 0, 'rescued_pass': 0}
    for f in Path(strict_dir, 'records').glob('*.json'):
        r = json.loads(f.read_text())
        if r['request_id'] in rescued:
            rr = json.loads(Path(rescue_dir, 'records', rescued[r['request_id']] + '.json').read_text())
            r = {**rr, 'request_id': r['request_id'], 'lenient_rescued': True, 'strict_record': r['request_id']}
            n['rescued'] += 1; n['rescued_pass'] += bool(r['binary_success'])
        else:
            n['strict'] += 1
        (out/f.name).write_text(json.dumps(r) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(n))


if __name__ == '__main__':
    {'agent': agent, 'single': single, 'merge': merge}[sys.argv[1]](*sys.argv[2:])
