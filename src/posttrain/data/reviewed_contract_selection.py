"""Release contracts must be frozen against an individually reviewed statement.

Generic compiler output remains a diagnostic proposal. Neither a heuristic
coverage label nor passing sample inputs authorizes deletion by itself.
"""
from copy import deepcopy
import re
from posttrain.common import digest


def _expression_lower_bound(value, domains):
    result=value['constant']
    for name,coefficient in value['terms'].items():
        lower,upper=domains.get(name,(None,None))
        endpoint=lower if coefficient>0 else upper
        if endpoint is None:return None
        result += coefficient*endpoint
    return result


def _expression_upper_bound(value, domains):
    result=value['constant']
    for name,coefficient in value['terms'].items():
        lower,upper=domains.get(name,(None,None))
        endpoint=upper if coefficient>0 else lower
        if endpoint is None:return None
        result += coefficient*endpoint
    return result


def _integer_domain(field, domains):
    lower_bounds=[];upper_bounds=[]
    for bound in field.get('bounds',[]):
        if bound.get('min_expr') is not None:
            value=_expression_lower_bound(bound['min_expr'],domains)
            if value is not None:lower_bounds.append(value)
        if bound.get('max_expr') is not None:
            value=_expression_upper_bound(bound['max_expr'],domains)
            if value is not None:upper_bounds.append(value)
    return (max(lower_bounds) if lower_bounds else None,
            min(upper_bounds) if upper_bounds else None)


def _complete_string_fields_are_token_safe(contract):
    domains={}
    for operation in contract.get('operations',[]):
        local_domains=domains.copy()
        target_domains=local_domains if operation.get('kind')=='repeat' else domains
        for field in operation.get('fields',[]):
            if field.get('kind')=='string':
                alphabet=field.get('alphabet')
                if not isinstance(alphabet,str) or not alphabet or any(c.isspace() for c in alphabet):
                    return False
                length=field.get('length_expr')
                if length is None:length=field.get('min_length_expr')
                if length is None:return False
                minimum=_expression_lower_bound(length,target_domains)
                if minimum is None or minimum<1:return False
                if field.get('length_binding'):
                    upper=field.get('length_expr',field.get('max_length_expr'))
                    maximum=_expression_upper_bound(upper,target_domains) if upper else None
                    target_domains[field['length_binding']]=(minimum,maximum)
            elif field.get('kind')=='int' and field.get('name'):
                target_domains[field['name']]=_integer_domain(field,target_domains)
    return True


def _schema(contract):
    def require(ok):
        if not ok:raise ValueError('reviewed_contract_schema_unsupported_or_malformed')
    def expression(value):
        require(isinstance(value,dict) and set(value)=={'constant','terms'})
        require(type(value['constant']) is int and isinstance(value['terms'],dict))
        require(all(isinstance(k,str) and k and type(v) is int for k,v in value['terms'].items()))
    adjudications=contract.get('reviewed_invalid_inputs',[])
    require(isinstance(adjudications,list))
    if adjudications:
        require(contract.get('coverage')=='partial')
        seen=set()
        for item in adjudications:
            require(isinstance(item,dict) and set(item)=={'stdin_sha256','reason','evidence','review_sha256'})
            for key in ('stdin_sha256','review_sha256'):
                require(isinstance(item.get(key),str) and re.fullmatch(r'[0-9a-f]{64}',item[key]) is not None)
            require(item['stdin_sha256'] not in seen);seen.add(item['stdin_sha256'])
            require(isinstance(item['reason'],str) and bool(item['reason']))
            require(isinstance(item['evidence'],list) and bool(item['evidence']))
    if contract.get('schema_version') == 2:
        require(set(contract) <= {'schema_version','coverage','layout_complete','limitations',
            'sound_for_deletion','header','case','count_var','case_sums','evidence',
            'output_review_holds','compiler','release_review_sha256','reviewed_invalid_inputs'})
        require(contract.get('coverage') in {'complete','partial'})
        require(type(contract.get('layout_complete')) is bool)
        require(isinstance(contract.get('evidence'),list) and bool(contract['evidence']))
        if contract['coverage'] == 'complete':
            require(contract['layout_complete'] and not contract.get('limitations'))
        for name in ('header','case'):
            child=contract.get(name)
            require(isinstance(child,dict) and child.get('schema_version')==1)
            require(child.get('coverage') in {'complete','partial'})
            require(not child.get('output_review_holds'))
            require(not child.get('reviewed_invalid_inputs'))
            _schema(child)
            require(child.get('layout_complete') is True)
            # Parsing must consume a fixed nonempty prefix on every case.
            require(child['operations'][0]['kind']=='fields')
            if contract['coverage']=='complete':require(child['coverage']=='complete')
        header=contract['header']
        count_var=contract.get('count_var')
        require(isinstance(count_var,str) and bool(count_var))
        require(any(op['kind']=='fields' and any(f.get('name')==count_var and f['kind']=='int'
                    for f in op['fields']) for op in header['operations']))
        sums=contract.get('case_sums',[])
        require(isinstance(sums,list))
        for rule in sums:
            require(set(rule)=={'id','expression','limit','operator','evidence'})
            require(isinstance(rule['id'],str) and bool(rule['id']))
            expression(rule['expression']);require(type(rule['limit']) is int)
            require(rule['operator'] in {'<','<=','≤','\\le','\\leq'})
            require(isinstance(rule['evidence'],list) and bool(rule['evidence']))
        for hold in contract.get('output_review_holds',[]):
            require(isinstance(hold,dict) and set(hold)=={'reason','evidence'})
            require(isinstance(hold['reason'],str) and bool(hold['reason']))
            require(isinstance(hold['evidence'],list) and bool(hold['evidence']))
        return
    require(contract.get('schema_version')==1)
    if 'layout_complete' in contract:require(type(contract['layout_complete']) is bool)
    if contract.get('coverage')=='complete':
        require(not contract.get('limitations'))
        require(contract.get('layout_complete') is not False)
    holds=contract.get('output_review_holds',[])
    require(isinstance(holds,list))
    for hold in holds:
        require(isinstance(hold,dict) and set(hold)=={'reason','evidence'})
        require(isinstance(hold['reason'],str) and bool(hold['reason']))
        require(isinstance(hold['evidence'],list) and bool(hold['evidence']))
    operations=contract.get('operations')
    require(isinstance(operations,list) and bool(operations))
    ids=set();bounds=[]
    for op in operations:
        require(op.get('kind') in {'fields','repeat'})
        require(isinstance(op.get('id'),str) and op['id'] not in ids);ids.add(op['id'])
        require(set(op)<={'id','kind','fields','evidence','count_expr','rows','array','distinct'})
        if op['kind']=='repeat':expression(op.get('count_expr'))
        require(isinstance(op.get('fields'),list) and bool(op['fields']))
        if op.get('distinct') and contract['coverage']=='complete':
            require(any(s.get('kind') in {'permutation','all_distinct'} and s.get('operation_id')==op['id']
                        and len(op['fields'])==1
                        for s in contract.get('semantics',[])))
        for field in op['fields']:
            require(field.get('kind') in {'int','string'})
            require(set(field)<={'kind','name','bounds','length_expr','evidence',
                                'alphabet','min_length_expr','max_length_expr','length_binding',
                                'contains_all','is_palindrome','starts_with','ends_with',
                                'distinct_characters','subsequence_of','character_counts',
                                'character_count_moduli','forbidden_prefixes','alphabet_prefix_length_expr'})
            if 'alphabet_prefix_length_expr' in field:
                require(field['kind']=='string' and bool(field.get('alphabet')))
                require(len(set(field['alphabet']))==len(field['alphabet']))
                expression(field['alphabet_prefix_length_expr'])
            if 'forbidden_prefixes' in field:
                require(field['kind']=='string' and isinstance(field['forbidden_prefixes'],list))
                require(all(isinstance(p,str) and p and set(p)<=set(field.get('alphabet',''))
                            for p in field['forbidden_prefixes']))
            for key in ('character_counts','character_count_moduli'):
                if key not in field:continue
                require(field['kind']=='string' and isinstance(field[key],dict))
                for char,value in field[key].items():
                    require(isinstance(char,str) and len(char)==1 and char in field.get('alphabet',''))
                    if key=='character_count_moduli':require(type(value) is int and value>0)
                    else:
                        require(isinstance(value,dict) and value and set(value)<={'min','max'})
                        require(all(type(v) is int and v>=0 for v in value.values()))
                        if 'max' in value:require(value.get('min',0)<=value['max'])
            for key in ('contains_all','starts_with','ends_with'):
                if key in field:
                    require(field['kind']=='string' and isinstance(field[key],str) and bool(field[key]))
                    require(set(field[key])<=set(field.get('alphabet','')))
            for key in ('is_palindrome','distinct_characters'):
                if key in field:require(field['kind']=='string' and type(field[key]) is bool)
            for key in ('length_binding','subsequence_of'):
                if key in field:
                    require(field['kind']=='string' and isinstance(field[key],str) and bool(field[key]))
                    require(op['kind']=='fields')
            if 'alphabet' in field:
                require(field['kind']=='string' and isinstance(field['alphabet'],str)
                        and bool(field['alphabet']) and not any(c.isspace() for c in field['alphabet']))
            for key in ('min_length_expr','max_length_expr'):
                if field.get(key) is not None:
                    require(field['kind']=='string');expression(field[key])
            if field.get('length_expr') is not None:
                require(field['kind']=='string');expression(field['length_expr'])
            if field.get('bounds'):require(field['kind']=='int')
            for bound in field.get('bounds',[]):
                require(set(bound)<={'min_expr','max_expr','evidence'})
                for name in ('min_expr','max_expr'):
                    if bound.get(name) is not None:expression(bound[name])
                bounds.append((field.get('name'),bound.get('min_expr'),bound.get('max_expr')))
    for rule in contract.get('constraints',[]):
        require(rule.get('kind') in {'range','upper_bound','product_upper_bound'})
        if rule['kind']=='range':
            require((rule.get('field_name'),rule.get('min_expr'),rule.get('max_expr')) in bounds)
        elif rule['kind']=='product_upper_bound':
            require(set(rule)<={'id','kind','factors','limit','evidence'})
            require(isinstance(rule.get('factors'),list) and len(rule['factors'])>=2)
            for factor in rule['factors']:expression(factor)
            require(type(rule.get('limit')) is int)
        else:
            expression(rule.get('expression'));require(type(rule.get('limit')) is int)
            require(rule.get('operator','≤') in {'≤','<=','\\le','\\leq','<'})
    for rule in contract.get('semantics',[]):
        require(rule.get('kind') in {'permutation','tree','all_distinct','ordered','excludes_value',
            'row_pairwise_neq','unique_tuple_rows','row_gcd_one','array_sum_upper_bound',
            'array_sum_lower_bound','array_sum_equals','aggregate_gcd_one','multiple_of_constant',
            'row_sum_lower_bound','row_sum_upper_bound','unique_unordered_pair_rows','row_tuple_neq','row_linear_upper_bound'})
        require(rule.get('operation_id') in ids)
        op=next(o for o in operations if o['id']==rule['operation_id'])
        if rule['kind']=='row_linear_upper_bound':
            require(set(rule)=={'id','kind','operation_id','evidence','expression','limit'})
            require(all(f['kind']=='int' for f in op['fields']))
            require(op['kind']=='fields' or bool(op.get('rows') or op.get('array')))
            expression(rule['expression']);require(type(rule['limit']) is int)
            continue
        if rule['kind'] in {'row_pairwise_neq','unique_tuple_rows','row_gcd_one',
                            'array_sum_upper_bound','array_sum_lower_bound','array_sum_equals',
                            'aggregate_gcd_one','multiple_of_constant','row_sum_lower_bound',
                            'row_sum_upper_bound','unique_unordered_pair_rows','row_tuple_neq'}:
            allowed={'id','kind','operation_id','evidence'}
            if rule['kind']=='row_tuple_neq':
                allowed.add('field_groups');groups=rule.get('field_groups')
                require(isinstance(groups,list) and len(groups)==2 and all(isinstance(g,list) and g for g in groups))
                require(len(groups[0])==len(groups[1]));names=[n for g in groups for n in g]
                require(all(isinstance(n,str) for n in names) and len(set(names))==len(names))
            elif rule['kind'] in {'row_pairwise_neq','unique_tuple_rows','row_gcd_one',
                                 'row_sum_lower_bound','row_sum_upper_bound','unique_unordered_pair_rows'}:
                allowed.add('field_names')
                names=rule.get('field_names')
                require(isinstance(names,list) and len(names)>=2 and all(isinstance(n,str) for n in names))
                require(len(set(names))==len(names))
                if rule['kind']=='unique_unordered_pair_rows':require(len(names)==2)
                if rule['kind'].startswith('row_sum_'):
                    allowed.add('expression');expression(rule.get('expression'))
            else:
                allowed.add('field_name');names=[rule.get('field_name')]
                if rule['kind'].startswith('array_sum_'):
                    allowed.add('expression');expression(rule.get('expression'))
                elif rule['kind']=='multiple_of_constant':
                    allowed.add('divisor');require(type(rule.get('divisor')) is int and rule['divisor']>0)
            require(set(rule)<=allowed)
            if op['kind']=='fields':
                require(rule['kind'] in {'row_pairwise_neq','row_gcd_one','multiple_of_constant',
                                        'row_sum_lower_bound','row_sum_upper_bound','row_tuple_neq'})
            else:require(bool(op.get('rows') or op.get('array')))
            require(all(any(f.get('name')==n and f['kind']=='int' for f in op['fields']) for n in names))
            continue
        require(op['kind']=='repeat')
        if rule['kind'] in {'permutation','all_distinct','ordered','excludes_value'}:
            require(rule.get('field_name') in {f.get('name') for f in op['fields']})
            require(bool(op.get('array') or op.get('rows')))
            require(next(f for f in op['fields'] if f.get('name')==rule['field_name'])['kind']=='int')
            if rule['kind']=='ordered':require(rule.get('operator') in {'lt','le','gt','ge'})
            if rule['kind']=='excludes_value':expression(rule.get('expression'))
        else:
            require(op.get('rows') is True and len(op['fields'])>=2)
            require(isinstance(rule.get('node_count_var'),str))
    if contract.get('coverage')=='complete':
        require(_complete_string_fields_are_token_safe(contract))


def select_reviewed_contract(statement, reviewed):
    entry=reviewed.get(digest(statement))
    if entry is None:
        return {'schema_version':1,'coverage':'unsupported','operations':[],
                'constraints':[],'semantics':[],
                'limitations':['no_statement_bound_release_review'],
                'sound_for_deletion':False,'compiler':'release_review_gate_v1'}
    contract=deepcopy(entry['contract'])
    if not entry.get('review_basis') or contract.get('sound_for_deletion') is not True:
        raise ValueError('reviewed_contract_not_authorized')
    if contract.get('coverage') not in {'partial','complete'}:
        raise ValueError('reviewed_contract_invalid_coverage')
    _schema(contract)
    evidence=[]
    def gather(node):
        evidence.extend(node.get('evidence',[]))
        for item in node.get('reviewed_invalid_inputs',[]):
            evidence.extend(item['evidence'])
        for hold in node.get('output_review_holds',[]):
            evidence.extend(hold['evidence'])
        for operation in node.get('operations',[]):
            evidence.extend(operation.get('evidence',[]))
            for field in operation.get('fields',[]):
                evidence.extend(field.get('evidence',[]))
                for bound in field.get('bounds',[]):evidence.extend(bound.get('evidence',[]))
        for rule in node.get('constraints',[])+node.get('semantics',[])+node.get('case_sums',[]):
            evidence.extend(rule.get('evidence',[]))
        if node.get('schema_version')==2:
            gather(node['header']);gather(node['case'])
    gather(contract)
    if not evidence:raise ValueError('reviewed_contract_no_evidence')
    for item in evidence:
        start,end=item.get('start'),item.get('end')
        if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(statement)
                or statement[start:end]!=item.get('text')):
            raise ValueError('reviewed_contract_evidence_mismatch')
    contract['release_review_sha256']=digest(entry)
    return contract
