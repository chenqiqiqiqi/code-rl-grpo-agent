from dataclasses import replace

import pytest

from posttrain.common import canonical
from posttrain.data.dedup import cluster_problems, exclude_final_clusters, source_key
from posttrain.data.sampling import largest_remainder, split_clusters, stratified_sample
from posttrain.data.schema import DataError, TestCase as Case, clean_tests
from posttrain.data.taco import from_taco


def test_empty_and_conflicting_tests():
    with pytest.raises(DataError,match='empty_tests'):
        clean_tests(())
    with pytest.raises(DataError,match='conflicting_tests'):
        clean_tests((Case('1','2'),Case('1','3')))
    assert len(clean_tests((Case('1','2\n'),Case('1',' 2 '))))==1


def test_hidden_data_not_in_prompt(problem):
    p=replace(problem,tests=(Case('hidden input 8fcb','hidden answer fa90'),),references=('secret solution b099',))
    prompt=canonical(p.prompt())
    assert all(value not in prompt for value in ('8fcb','fa90','b099','references','tests'))


def test_reference_and_test_schema(problem):
    with pytest.raises(DataError):
        replace(problem,references=())
    with pytest.raises(DataError):
        Case([1],2)
    with pytest.raises(DataError):
        replace(problem,time_limit_s=float('nan'))


def test_taco_json_only_and_interface_filter():
    row={'question':'Print n','solutions':'["print(input())"]','input_output':'{"inputs":["1"],"outputs":["1"]}'}
    assert from_taco(row,'frozen',1,min_tests=1).tests[0].stdin=='1'
    for value in ('__import__("os").system("echo bad")','{"fn_name":"solve"}','{"inputs":[1],"outputs":[1]}'):
        with pytest.raises(DataError):
            from_taco({**row,'input_output':value},'frozen',1,min_tests=1)


def test_source_transitive_dedup_and_final_exclusion(problem):
    a=replace(problem,source_id='https://codeforces.com/contest/123/problem/A')
    b=replace(problem,problem_id='p2',source_id='https://codeforces.com/problemset/problem/123/A',statement='Different wording')
    final=replace(problem,problem_id='final',source_id='another',statement=b.statement)
    other=replace(problem,problem_id='other',source_id='distinct',statement='Find minimum spanning tree of weighted graph')
    result,links=cluster_problems([a,b,final,other])
    assert len({p.cluster_id for p in result[:3]})==1
    kept,removed=exclude_final_clusters(result,{'final'})
    assert [p.problem_id for p in kept]==['other']
    assert set(removed)=={'p1','p2'}
    assert source_key(a.source_id)==source_key(b.source_id)


def test_split_cluster_isolation_and_determinism(problem):
    items=[replace(problem,problem_id=f'p{i}',cluster_id=f'c{i//2}') for i in range(40)]
    split=split_clusters(items)
    assert split==split_clusters(list(reversed(items)))
    train={split['cluster_by_id'][k] for k in split['train_ids']}
    dev={split['cluster_by_id'][k] for k in split['dev_ids']}
    assert train.isdisjoint(dev)


def test_largest_remainder_and_frozen_sampling():
    assert largest_remainder({'a':1,'b':1,'c':1},2)=={'a':1,'b':1,'c':0}
    with pytest.raises(ValueError):
        largest_remainder({'easy':199},200)
    rows=[{'task_id':str(i),'difficulty':['easy','medium','hard'][i%3]} for i in range(350)]
    sample=stratified_sample(rows,200)
    assert len(sample)==len({x['task_id'] for x in sample})==200
    assert sample==stratified_sample(list(reversed(rows)),200)
