from posttrain.data.constraint_engine import compile_contract,validate_input


def test_leading_unnamed_test_count_is_not_bound_to_later_pair():
    text='''Input format:
The first line contains the number of test cases. Followed by two integers A and B. Both A and B are inclusive.
Constraints:
1 <= Test Cases <= 1000
1 <= B <= 10^6
1 <= A < B
Output
Print a number.
'''
    c=compile_contract(text)
    assert c['coverage']!='complete'
    assert validate_input(c,'500\n14 20\n')['violations']==[]


def test_leading_unnamed_test_count_is_not_bound_to_case_scalar():
    text='''Input
First line contains number of testcases 1<t<1000, for each test case only one integer K is given. (0<K<75).
Output
Print the answer.
'''
    c=compile_contract(text)
    assert c['coverage']!='complete'
    assert validate_input(c,'126\n2\n')['violations']==[]


def test_explicit_first_scalar_still_checks_its_own_bound():
    c=compile_contract('Input\nThe first line contains a single integer t (1 <= t <= 10).\nOutput\nPrint t.')
    assert validate_input(c,'11')['violations'][0]['reason']=='integer_out_of_range'


def test_latex_grouped_upper_bound_is_not_truncated():
    c=compile_contract(r'Input'+'\n'+r'The first line contains one integer n (1 \le n \le 1\,000).'+'\nOutput\nPrint n.')
    assert validate_input(c,'900')['violations']==[]
    assert validate_input(c,'1001')['violations']


def test_heterogeneous_repeated_rows_cannot_drop_string_tokens():
    c=compile_contract('Input\nThe first line contains one integer n (1 <= n <= 1000).\nThe next n lines contain one integer c (1 <= c <= 100000) and a string s.\nOutput\nPrint n.')
    assert validate_input(c,'2\n5 ABC\n6 BC')['violations']==[]
    assert c['coverage']!='complete'


def test_compound_string_length_never_uses_only_variable_prefix():
    c=compile_contract('Input\nThe first line contains one integer m (1 <= m <= 100).\nThe second line contains a string s of length m+2.\nOutput\nPrint s.')
    assert validate_input(c,'2\n0000')['violations']==[]


def test_mixed_direction_bounds_do_not_silently_drop_one_side():
    c=compile_contract('Input\nThe first line contains one integer x (1 <= x >= 0).\nOutput\nPrint x.')
    assert validate_input(c,'0')['violations'] or c['coverage']!='complete'


def test_unknown_repeat_length_is_not_a_failed_permutation():
    c={'coverage':'partial','operations':[{'id':'items','kind':'repeat',
       'count_expr':{'constant':0,'terms':{'unknown_n':1}},'array':True,
       'fields':[{'name':'a','kind':'int'}]}],
       'semantics':[{'id':'perm','kind':'permutation','operation_id':'items','field_name':'a','evidence':[]}]}
    result=validate_input(c,'1 2 3')
    assert result == {'coverage':'partial','violations':[]}


def test_unparsed_edges_are_not_a_failed_tree():
    c={'coverage':'partial','operations':[
       {'id':'header','kind':'fields','fields':[{'name':'n','kind':'int'}]},
       {'id':'edges','kind':'repeat','count_expr':{'constant':0,'terms':{'unknown_m':1}},
        'rows':True,'fields':[{'name':'u','kind':'int'},{'name':'v','kind':'int'}]}],
       'semantics':[{'id':'tree','kind':'tree','operation_id':'edges','node_count_var':'n','evidence':[]}]}
    assert validate_input(c,'3 1 2 2 3') == {'coverage':'partial','violations':[]}


def test_unresolved_string_length_cannot_retain_complete_coverage():
    c={'coverage':'complete','operations':[{'id':'s','kind':'fields','fields':[
       {'kind':'string','name':'s','length_expr':{'constant':0,'terms':{'unknown_n':1}}}]}]}
    assert validate_input(c,'abc') == {'coverage':'partial','violations':[]}
