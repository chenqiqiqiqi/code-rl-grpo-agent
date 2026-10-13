from posttrain.data.checker_compatibility import checker_review_flags


def test_case_and_construction_variants_need_checker_review():
    for text in ('You may print the answer in any case.',
                 'You can print each letter in arbitrary case (small or large).',
                 'Print YES and any correct coloring.',
                 'It is not required to minimize k.',
                 'Construct any one N-by-N matrix a that satisfies the conditions.',
                 'Print a solution in the following format:',
                 'Print your solution in the following format:',
                 'Any of them will be accepted.',
                 'If there are multiple pairs that maximize the value, any of them is accepted.',
                 'Choose one such integer and print it.',
                 'The output will be judged correct if the graph satisfies the conditions.',
                 'The output is correct when the relative difference is not greater than 10^{-9}.',
                 'The absolute or relative error should not exceed 1e-6.',
                 'All numbers should be correct up to 10^{-6} relative or absolute precision.',
                 "You don't need to minimize the number of moves."):
        flags=checker_review_flags(text)
        assert flags and all(text[f['start']:f['end']]==f['evidence'] for f in flags)


def test_deterministic_minimum_and_tie_break_are_not_flagged():
    assert checker_review_flags('Print the minimum number of moves. If tied, print the smallest value.')==[]
    assert checker_review_flags('Print the minimum absolute difference between the two integers.')==[]
