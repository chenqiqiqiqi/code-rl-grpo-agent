"""Static, statement-bound input checks. No submitted program is executed."""
import re
from posttrain.common import digest
from posttrain.data.review_contracts import check_input


def check_contract(rule, text):
    tokens=text.split()
    if len(tokens)!=1:
        return 'expected_one_token'
    token=tokens[0]
    if rule['kind']=='integer':
        if not re.fullmatch(r'[+-]?[0-9]+',token):
            return 'expected_ascii_integer'
        # Avoid unbounded Python integer conversion of malformed huge input.
        digits=token.lstrip('+-').lstrip('0') or '0'
        if len(digits)>max(len(str(abs(rule['min']))),len(str(abs(rule['max'])))):
            return 'integer_out_of_range'
        value=int(('-' if token.startswith('-') else '')+digits)
        if not rule['min']<=value<=rule['max']:
            return 'integer_out_of_range'
    elif rule['kind']=='string':
        if not rule['min']<=len(token)<=rule['max']:
            return 'string_length_out_of_range'
        if set(token)-set(rule['alphabet']):
            return 'string_alphabet'
        if rule.get('adjacent_distinct') and any(a==b for a,b in zip(token,token[1:])):
            return 'equal_adjacent_characters'
    else:
        raise ValueError('unsupported_contract_kind')
    return None


def audit_problem(row, contracts):
    result={'problem_id':row['problem_id'],'statement_sha256':digest(row['statement']),
            'tests_sha256':digest(row['tests']),'tests_available':len(row['tests']),
            'tests_checked':0,'violations':[],'status':'unsupported'}
    rule=contracts.get(row['problem_id'])
    if rule is None:return result
    if rule['statement_sha256']!=result['statement_sha256']:
        raise ValueError('statement_changed:'+row['problem_id'])
    result['contract_sha256']=digest(rule)
    if rule['kind']=='statement_quarantine':
        result.update(status='statement_quarantine',reason=rule['reason'])
        return result
    if not row['tests']:raise ValueError('empty_tests')
    for i,test in enumerate(row['tests']):
        error=(check_input(row['problem_id'],test['stdin']) if rule['kind']=='legacy_review'
               else check_contract(rule,test['stdin']))
        result['tests_checked']+=1
        if error:result['violations'].append({'test_index':i,'stdin_sha256':digest(test['stdin']),'reason':error})
    result['status']='invalid_input' if result['violations'] else 'input_domain_pass'
    return result
