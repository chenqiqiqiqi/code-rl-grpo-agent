import pytest

from posttrain.data.schema import Problem, TestCase as Case


@pytest.fixture
def problem():
    return Problem('p1','synthetic','v1','s:1','Read n and print n plus one.',
                   tuple(Case(str(n)+'\n',str(n+1)+'\n') for n in range(5)),
                   ('print(int(input())+1)',),difficulty='easy')
