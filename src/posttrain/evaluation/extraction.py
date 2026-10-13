from __future__ import annotations

import ast
import re


def extract_python(text: str) -> str | None:
    """One Python fence or raw parseable code; do not guess among many blocks."""
    fences=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,flags=re.S|re.I)
    if len(fences)>1:
        return None
    code=(fences[0] if fences else text).strip()
    if not code:
        return None
    try:
        ast.parse(code)
    except (SyntaxError,ValueError,RecursionError):
        return None
    return code+'\n'
