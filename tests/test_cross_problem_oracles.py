from posttrain.data.cross_problem_oracles import find_oracle_conflicts


def row(pid, statement, tests, cluster='c'):
    return {'problem_id':pid,'statement':statement,'cluster_id':cluster,
            'tests':[{'stdin':i,'stdout':o} for i,o in tests]}


def test_same_task_and_input_disagreeing_answers_are_all_flagged():
    report=find_oracle_conflicts(iter([
        row('a','Task',[('1\n','2')]),row('b','Task',[('1\r\n','3')])]))
    assert report['affected_problem_ids']==['a','b']
    assert report['conflicting_input_groups']==1
    assert report['conflicts'][0]['members'][0]['test_index']==0


def test_different_tasks_or_whitespace_equivalent_outputs_are_not_conflicts():
    report=find_oracle_conflicts(iter([
        row('a','Task',[('1','2 3')]),row('b','Task',[('1','2\n3')]),
        row('c','Different task',[('1','4')])]))
    assert report['conflicting_input_groups']==0


def test_input_spaces_remain_semantically_distinct():
    report=find_oracle_conflicts(iter([
        row('a','Task',[('a b','2')]),row('b','Task',[('a  b','3')])]))
    assert report['affected_problem_ids']==[]
