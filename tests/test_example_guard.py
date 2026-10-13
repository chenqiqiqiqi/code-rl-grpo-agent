from posttrain.data.example_guard import statement_example_inputs


def test_does_not_treat_input_description_as_a_sample():
    text='Input\nA positive integer n.\nOutput\nPrint n.\nExamples\nInput\n3\nOutput\n3\nInput\n4\nOutput\n4'
    assert statement_example_inputs(text)==['3\n','4\n']


def test_sample_headers_and_numbering():
    text='Task\n-----Sample Input 1-----\n2 3\n-----Sample Output 1-----\n5\n'
    assert statement_example_inputs(text)==['2 3\n']


def test_no_guess_for_inline_or_unclosed_samples():
    assert statement_example_inputs('Input: n. Output: n.')==[]
    assert statement_example_inputs('Example\nInput\n3')==[]


def test_preserve_space_sensitive_sample_lines():
    text='Examples\nInput\n a b \n  c\nOutput\nX'
    assert statement_example_inputs(text)==[' a b \n  c\n']
