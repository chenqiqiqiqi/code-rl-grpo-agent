"""Score train-split candidates on the isolated VM executor (binary fail-fast, full frozen tests).

Usage (inside VM project root): python3 score_train_candidates.py <candidates.jsonl> <out_dir> [--sequential-g]
--sequential-g: for g>1, score candidate j only if candidates < j of the same problem all failed
(finds the first passing sample per problem with less execution).
"""
import sys, json, time
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src'))
from posttrain.common import digest, write_json
from posttrain.data.schema import Problem
from posttrain.execution.contracts import Limits
from posttrain.execution.docker_backend import DockerBackend
from posttrain.execution.judge import Judge
from posttrain.evaluation.extraction import extract_python

IMAGE = 'sha256:0106a6090782e770c43597f20b3e579584c575b1166d3a5dad969f18d4fe5e56'
ATT = ROOT / 'runs/isolation-stage4-20261001b/attestation.json'


def main():
    cand_path, out = Path(sys.argv[1]), Path(sys.argv[2]); sequential = '--sequential-g' in sys.argv
    source = ROOT / ('cpu-data/dev-reference-passed.jsonl' if '--dev' in sys.argv else 'cpu-data/train-reference-passed.jsonl')
    problems = {p.problem_id: p for p in map(Problem.from_dict, map(json.loads, source.read_text().splitlines()))}
    cands = [json.loads(l) for l in cand_path.read_text().splitlines()]
    out.mkdir(parents=True, exist_ok=True); (out / 'records').mkdir(exist_ok=True)
    backend = DockerBackend(IMAGE, Limits(time_s=30), ATT); start = time.monotonic()

    def score(c):
        path = out / 'records' / (c['request_id'] + '.json')
        if path.exists():
            r = json.loads(path.read_text())
            if r.get('state') == 'complete': return r
        p = problems[c['problem_id']]
        assert c['tests_hash'] == p.tests_hash and c['test_count'] == len(p.tests)
        code = extract_python(c['text']); reject = code is None
        if reject: code = 'def <POSTTRAIN_FORMAT_REJECTED>(\n'
        mode = 'full' if '--full' in sys.argv else 'binary_fail_fast'
        for attempt in range(3):  # transient infrastructure failures are retried, never scored as 0
            result = Judge(backend).score(code, p.tests, Limits(time_s=p.time_limit_s), c['request_id'] + ('' if not attempt else f'-r{attempt}'), scoring_mode=mode)
            if (result.valid if mode == 'full' else result.binary_valid): break
            print(json.dumps({'retry_invalid': c['request_id'], 'attempt': attempt}), flush=True); time.sleep(5)
        else:
            raise RuntimeError('invalid_' + c['request_id'])
        row = {'state': 'complete', 'problem_id': p.problem_id, 'request_id': c['request_id'], 'candidate_index': c['candidate_index'],
               'format_rejected': reject, 'finish_reason': c['finish_reason'], 'output_tokens': c['output_tokens'],
               'binary_success': result.passed == result.total and not reject, 'passed_tests': result.passed, 'total_tests': result.total, 'status': getattr(result.status, 'value', str(result.status)),
               'limits': asdict(Limits(time_s=p.time_limit_s)), 'executor_version': backend.version, 'code_sha256': digest(code),
               'text_sha256': digest(c['text']), 'result': result.to_dict()}
        write_json(path, row); return row

    rows = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        if not sequential:
            rows = list(pool.map(score, cands))
        else:
            by = {}
            for c in cands: by.setdefault(c['problem_id'], []).append(c)
            g = max(len(v) for v in by.values()); solved = set()
            for j in range(g):
                todo = [v[j] for pid, v in by.items() if pid not in solved and j < len(v)]
                res = list(pool.map(score, todo)); rows += res
                solved |= {r['problem_id'] for r in res if r['binary_success']}
                print(json.dumps({'round': j, 'scored': len(todo), 'solved_total': len(solved), 'elapsed_s': round(time.monotonic() - start)}), flush=True)
    summary = {'candidates_file_sha256': digest(cand_path.read_text()), 'scored': len(rows),
               'passed_candidates': sum(r['binary_success'] for r in rows),
               'problems_solved': len({r['problem_id'] for r in rows if r['binary_success']}),
               'elapsed_s': time.monotonic() - start, 'executor_version': backend.version, 'sequential_g': sequential}
    write_json(out / 'summary.json', summary); print(json.dumps(summary))


if __name__ == '__main__':
    main()
