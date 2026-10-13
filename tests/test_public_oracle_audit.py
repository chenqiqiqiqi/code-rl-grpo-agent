from posttrain.data.public_oracle_audit import audit_public_answers,public_numeric_examples


def test_explicit_sample_output_mismatch_is_only_review_flag():
    row={'problem_id':'x','statement':'Task\nExamples\nInput\n3\nOutput\n4\nNote\nSome prose.',
         'tests':[{'stdin':'3\n','stdout':'5'}]}
    r=audit_public_answers(row)
    assert len(r['conflicts'])==1
    assert r['action']=='review_only_no_test_or_answer_changed'


def test_output_prose_and_unrecognized_fences_are_not_oracles():
    assert public_numeric_examples('Input\nAn integer n\nOutput\n42')==[]
    assert public_numeric_examples('Examples\nInput\n3\nOutput\n4\nSome unknown prose.')==[]
    assert public_numeric_examples('Examples\nInput\n```\n3\n```\nOutput\n4')==[]


def test_internal_input_spaces_are_preserved_and_output_whitespace_normalized():
    row={'problem_id':'x','statement':'Task\nSample Input 1\na  b\nSample Output 1\n4 5',
         'tests':[{'stdin':'a b','stdout':'9'},{'stdin':'a  b','stdout':'4\n5'}]}
    r=audit_public_answers(row)
    assert r['matching_test_example_pairs']==1 and r['conflicts']==[]
