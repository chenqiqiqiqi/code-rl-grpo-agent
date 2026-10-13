from __future__ import annotations

import ast
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

from posttrain.common import digest
from posttrain.data.schema import Problem


@dataclass(frozen=True)
class PublicProblem:
    """Final benchmark dedup record: no tests, answers, or reference solution."""
    problem_id: str
    source_id: str
    statement: str
    references: tuple[str,...] = ()
    cluster_id: str = ''


def normalized_statement(text: str) -> str:
    return " ".join(re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFKC", text).casefold()))


def source_key(source_id: str) -> str:
    value = source_id.lower().strip().rstrip("/")
    if "://" in value:
        u = urlsplit(value)
        host = u.netloc.removeprefix("www.")
        path = u.path.rstrip("/")
        if host.endswith("codeforces.com"):
            m = re.search(r"/(?:contest|gym)/(\d+)/problem/(\w+)|/problemset/problem/(\d+)/(\w+)", path)
            if m:
                return f"codeforces:{m[1] or m[3]}:{m[2] or m[4]}"
        return host + path
    return value


def shingles(text: str, width: int = 5) -> frozenset[str]:
    words = normalized_statement(text).split()
    return frozenset(" ".join(words[i:i+width]) for i in range(max(1, len(words)-width+1)))


def code_fingerprint(code: str) -> str | None:
    try:
        return digest(ast.dump(ast.parse(code), include_attributes=False))
    except (SyntaxError, ValueError, RecursionError):
        return None


def cluster_problems(problems: list[Problem | PublicProblem], threshold: float = 0.8) -> tuple[list, list[dict]]:
    """Transitive union of provenance, exact statement, and Jaccard near-duplicates.

    A code match is only supporting evidence with >= 0.6 statement similarity.
    Do not merge unrelated easy tasks solely because their reference code matches.
    """
    if not 0.6 <= threshold <= 1:
        raise ValueError("invalid dedup threshold")
    if len({p.problem_id for p in problems}) != len(problems):
        raise ValueError("duplicate_problem_id")
    parent = list(range(len(problems)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        parent[find(j)] = find(i)
    seen_source, seen_text = {}, {}
    bags = [shingles(p.statement) for p in problems]
    codes = [{c for ref in p.references if (c := code_fingerprint(ref))} for p in problems]
    postings = defaultdict(set)
    audit = []
    for i,p in enumerate(problems):
        for kind,key,seen in (("source",source_key(p.source_id),seen_source), ("exact_text",normalized_statement(p.statement),seen_text)):
            if key in seen:
                j = seen[key]
                union(i,j)
                audit.append({"left":p.problem_id,"right":problems[j].problem_id,"reason":kind,"similarity":1.0})
            seen[key] = i
        overlaps = defaultdict(int)
        for token in bags[i]:
            for j in postings[token]:
                overlaps[j] += 1
        for j,n in sorted(overlaps.items()):
            score = n / len(bags[i] | bags[j])
            reason = "near_text" if score >= threshold else "text_and_code" if score >= .6 and codes[i] & codes[j] else None
            if reason:
                union(i,j)
                audit.append({"left":p.problem_id,"right":problems[j].problem_id,"reason":reason,"similarity":score})
        for token in bags[i]:
            postings[token].add(i)
    members = defaultdict(list)
    for i,p in enumerate(problems):
        members[find(i)].append(p.problem_id)
    ids = {root:"cluster-"+digest(sorted(group))[:20] for root,group in members.items()}
    return [replace(p,cluster_id=ids[find(i)]) for i,p in enumerate(problems)], audit


def exclude_final_clusters(problems: list[Problem], final_ids: set[str]) -> tuple[list[Problem], list[str]]:
    if not final_ids <= {p.problem_id for p in problems}:
        raise ValueError("final ID missing from joint dedup input")
    protected = {p.cluster_id for p in problems if p.problem_id in final_ids}
    if "" in protected:
        raise ValueError("unclustered_final")
    kept = [p for p in problems if p.cluster_id not in protected]
    removed = [p.problem_id for p in problems if p.problem_id not in final_ids and p.cluster_id in protected]
    return kept, removed
