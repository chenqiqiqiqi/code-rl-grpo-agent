"""Detect conflicting expected outputs across identical task statements.

Conflicts are review flags, not proof that either output is wrong: a task may
accept multiple answers. No choice of oracle is made and no tests are deleted.
"""
from posttrain.common import digest
from posttrain.data.schema import normalize_output


def find_oracle_conflicts(rows):
    groups={};seen=set();tests=0
    for row in rows:
        pid=row['problem_id']
        if pid in seen:raise ValueError('duplicate_problem_id')
        seen.add(pid);statement_hash=digest(row['statement'])
        for index,test in enumerate(row['tests']):
            tests+=1
            key=(statement_hash,digest(test['stdin'].replace('\r\n','\n')))
            output_hash=digest(normalize_output(test['stdout']))
            member={'problem_id':pid,'test_index':index,'output_sha256':output_hash}
            groups.setdefault(key,[]).append(member)
    conflicts=[];affected=set()
    for (statement_hash,input_hash),members in sorted(groups.items()):
        if len({m['output_sha256'] for m in members})<2:continue
        conflicts.append({'statement_sha256':statement_hash,'stdin_sha256':input_hash,
                          'members':members})
        affected.update(m['problem_id'] for m in members)
    return {'schema_version':1,'problems_checked':len(seen),'tests_checked':tests,
            'scope':'identical_statement_and_identical_input_except_CRLF',
            'conflicting_input_groups':len(conflicts),'affected_problem_ids':sorted(affected),
            'conflicts':conflicts,'action':'review_required_no_oracle_chosen_no_test_deleted'}
