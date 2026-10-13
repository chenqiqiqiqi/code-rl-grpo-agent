"""Read only explicitly delimited public example inputs for parser self-checks."""
import re

_HEADER=re.compile(r'(?im)^[ \t]*[-#=]*[ \t]*(examples?|(?:sample[ \t]+)?input(?:[ \t]+\d+)?|(?:sample[ \t]+)?output(?:[ \t]+\d+)?)[ \t]*[-#=]*[ \t]*\r?$',re.I)


def statement_example_inputs(statement):
    headers=list(_HEADER.finditer(statement));in_examples=False;result=[]
    for i,header in enumerate(headers):
        label=header.group(1).lower()
        if label in ('example','examples'):in_examples=True;continue
        if label.startswith('sample input'):in_examples=True
        if not in_examples or not re.fullmatch(r'(sample\s+)?input(?:\s+\d+)?',label):continue
        if i+1==len(headers):continue
        following=headers[i+1]
        if not re.fullmatch(r'(sample\s+)?output(?:\s+\d+)?',following.group(1).lower()):continue
        text=statement[header.end():following.start()].replace('\r\n','\n').strip('\n')
        # Fences/tables require a different parser: do not guess their contents.
        if not text or '```' in text or '<pre>' in text or '</pre>' in text:continue
        result.append(text+'\n')
    return result
