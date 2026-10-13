"""Diagnostic comparison with explicitly delimited public example answers.

Only numeric / YES-NO output blocks are recognized. Newline framing is ignored
for matching inputs; all other whitespace is preserved. Findings need review,
and are never a rule for rewriting expected answers or deleting tests.
"""
import re
from posttrain.common import digest
from posttrain.data.schema import normalize_output


_HEADER=re.compile(r'(?im)^[ \t]*[-#=]*[ \t]*(examples?|(?:sample[ \t]+)?input(?:[ \t]+\d+)?|(?:sample[ \t]+)?output(?:[ \t]+\d+)?|notes?|explanation)[ \t]*[-#=]*[ \t]*\r?$')


def public_numeric_examples(statement):
    headers=list(_HEADER.finditer(statement));active=False;result=[]
    for i,h in enumerate(headers):
        label=h.group(1).lower()
        if label in ('example','examples') or label.startswith('sample input'):active=True
        if not active or not re.fullmatch(r'(sample\s+)?input(?:\s+\d+)?',label):continue
        if i+1>=len(headers):continue
        out=headers[i+1]
        if not re.fullmatch(r'(sample\s+)?output(?:\s+\d+)?',out.group(1).lower()):continue
        end=headers[i+2].start() if i+2<len(headers) else len(statement)
        inp=statement[h.end():out.start()].replace('\r\n','\n').strip('\n')
        answer=statement[out.end():end].strip()
        tokens=answer.split()
        if not inp or not tokens or any(not re.fullmatch(r'[+-]?[0-9]+|[Yy][Ee][Ss]|[Nn][Oo]',t) for t in tokens):continue
        if any(marker in inp for marker in ('```','<pre>','</pre>')):continue
        result.append({'stdin':inp,'stdout':answer,'output_start':out.end(),'output_end':end})
    return result


def audit_public_answers(row):
    samples=public_numeric_examples(row['statement']);by_input={}
    for number,sample in enumerate(samples):
        by_input.setdefault(sample['stdin'],[]).append((number,sample))
    conflicts=[];matched=0
    for index,test in enumerate(row['tests']):
        key=test['stdin'].replace('\r\n','\n').strip('\n')
        for number,sample in by_input.get(key,[]):
            matched+=1
            if normalize_output(test['stdout'])!=normalize_output(sample['stdout']):
                conflicts.append({'test_index':index,'sample_index':number,'stdin_sha256':digest(test['stdin']),
                    'expected_output_sha256':digest(normalize_output(test['stdout'])),
                    'public_output_sha256':digest(normalize_output(sample['stdout'])),
                    'public_output_span':[sample['output_start'],sample['output_end']]})
    return {'problem_id':row['problem_id'],'statement_sha256':digest(row['statement']),
            'recognized_public_examples':len(samples),'matching_test_example_pairs':matched,
            'conflicts':conflicts,'action':'review_only_no_test_or_answer_changed'}
