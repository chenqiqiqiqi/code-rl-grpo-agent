"""Extra diagnostic flags for the frozen exact-token, case-sensitive checker."""
import re

_PATTERNS=(
    ('floating_point_tolerance',r'\b(?:absolute|relative)\s+(?:(?:or|and)\s+(?:absolute|relative)\s+)?(?:error|precision|accuracy)\b'),
    ('floating_point_tolerance',r'\brelative\s+difference\b'),
    ('case_insensitive_output',r'\b(?:print|output)\b[^.\n]{0,120}\b(?:any|arbitrary)\s+case\b'),
    ('case_insensitive_output',r'\b(?:answer|answers|output)\s+in\s+any\s+case\b'),
    ('non_unique_output',r'\bany\s+(?:correct\s+(?:coloring|colouring|answer|solution)|timetable|number\s+that\s+(?:meets|satisfies))\b'),
    ('non_unique_output',r'\b(?:find|construct|give)\s+any\s+(?:of\s+them|way|solution|answer)\b'),
    ('non_unique_output',r'\bconstruct\s+any\b'),
    ('non_unique_output',r'\bprint\s+(?:a|your)\s+solution\b'),
    ('non_unique_output',r'\bany\s+of\s+them\s+(?:will\s+be\s+accepted|is\s+(?:acceptable|accepted))\b'),
    ('non_unique_output',r'\bchoose\s+one\s+such\s+integer\s+and\s+print\b'),
    ('non_unique_output',r'\boutput\s+will\s+be\s+judged\s+correct\s+if\b'),
    ('non_unique_output',r'\b(?:not\s+required|don.t\s+(?:need|have)|do\s+not\s+(?:need|have))\s+to\s+minimize\s+(?:k\b|the\s+number\s+of\s+(?:moves|operations))'),
)


def checker_review_flags(statement):
    result=[]
    for reason,pattern in _PATTERNS:
        for match in re.finditer(pattern,statement,re.I):
            result.append({'reason':reason,'start':match.start(),'end':match.end(),
                           'evidence':match.group(0)})
    return result
