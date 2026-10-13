"""Conservative public-statement screen for the current exact-token comparator.

Absence of a match is not proof of a unique answer or complete statement; recorded
spot-checks remain necessary. No candidate outcomes or final benchmark scores are
used here. Flagged tasks need a checker/statement review before training inclusion.
"""
import re


VERSION='exact-stdio-eligibility-v2'
PATTERNS=(
    ('unresolved_image_placeholder',r'<image(?:\s[^>]*)?>|\[image\]|<img\b|!\[[^\]]*\]\('),
    ('non_unique_output',r'\b(?:print(?:ing)?|output|return)\s+(?:an?\s+)?any\b'),
    ('non_unique_output',r'\b(?:print|output|return)\b[^.\n]{0,160}\b(?:in any order|in arbitrary order)\b'),
    ('non_unique_output',r'\b(?:multiple|several|many)\s+(?:[\w-]+\s+){0,4}(?:answers|solutions|sequences|ways)[^.]{0,200}\bany\b'),
    ('non_unique_output',r'\b(?:show|output|print)\s+(?:one|an?)\s+(?:example|possible|valid|good|such|way)\b'),
    ('numeric_tolerance',r'\b(?:absolute(?:\s+or\s+relative)?|relative(?:\s+or\s+absolute)?)\s+error\b'),
    ('case_insensitive_checker_required',r'\bcase[- ]insensitive\b'),
    ('custom_checker_required',r'\b(?:special judge|output.only problem)\b'),
)


def statement_issues(statement):
    issues=[]
    for reason,pattern in PATTERNS:
        match=re.search(pattern,statement,re.I)
        if match and reason not in {r['reason'] for r in issues}:
            issues.append({'reason':reason,'excerpt':statement[max(0,match.start()-60):min(len(statement),match.end()+100)],
                'match_start':match.start(),'match_end':match.end()})
    return issues
