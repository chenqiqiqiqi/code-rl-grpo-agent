import pytest

from posttrain.data.schema import DataError
from posttrain.data.taco import from_taco


@pytest.mark.parametrize('statement,reason',[
    ('Use this missing formula: <image>. Print the result.','unresolved_image_placeholder'),
    ('The graph is shown in [Image]. Print its answer.','unresolved_image_placeholder'),
    ('Print any valid sequence.','non_unique_output'),
    ('If there are multiple solutions, print any of them.','non_unique_output'),
    ('Output the indices in any order.','non_unique_output'),
    ('If there are multiple good sequences with the same minimum sum, printing any of them is accepted.','non_unique_output'),
    ('If there are multiple ways that minimize the total cost, printing any one of them is accepted.','non_unique_output'),
    ('If it exists, show one example.','non_unique_output'),
    ('The judge is case-insensitive; yes and YES are accepted.','case_insensitive_checker_required'),
    ('Your answer is accepted if its absolute or relative error is at most 1e-6.','numeric_tolerance'),
])
def test_incompatible_questions_do_not_enter_exact_comparator_pool(statement,reason):
    row={'question':statement,'solutions':['print(1)'],'picture_num':0,
        'input_output':{'inputs':['1'],'outputs':['1']}}
    with pytest.raises(DataError,match=reason):from_taco(row,'fixture',0,min_tests=1)


def test_any_value_in_input_is_not_an_output_contract():
    row={'question':'The input can contain any integers. Print their sum.',
        'solutions':['print(1)'],'input_output':{'inputs':['1'],'outputs':['1']}}
    assert from_taco(row,'fixture',0,min_tests=1).problem_id=='taco-train-000000'
