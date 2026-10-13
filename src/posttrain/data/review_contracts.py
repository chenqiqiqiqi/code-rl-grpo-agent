"""Static input-domain checks for the 12 frozen AI-review samples only.

These parse data; they never run submitted/reference programs or validate outputs.
Unknown problems are explicitly unsupported, not assumed valid.
"""
from fractions import Fraction


IDS = {'000181','002585','003121','005860','003355','001870',
       '008466','009984','008845','005992','002803','004138'}


def check_input(problem_id, text):
    key = problem_id.removeprefix('taco-train-')
    if key not in IDS:
        raise ValueError('unsupported_review_contract')
    tokens = iter(text.split())
    def number(low, high):
        value = int(next(tokens))
        if not low <= value <= high:
            raise ValueError(f'integer_out_of_range:{value}:expected_{low}..{high}')
        return value
    def tree_edges(n, edges):
        parent = list(range(n))
        def find(a):
            while a != parent[a]:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a,b in edges:
            a,b = find(a),find(b)
            if a == b:
                raise ValueError('cycle_or_duplicate_edge')
            parent[a] = b
    try:
        if key in {'000181','003355','005992'}:
            number(*{'000181':(3,500),'003355':(1,100),'005992':(1,10**9)}[key])
        elif key == '002585':
            for _ in range(number(1,10**4)):
                left=number(1,200000);number(left,200000)
        elif key == '003121':
            total=0
            for _ in range(number(1,10000)):
                number(1,10**9);m=number(1,100000);total+=m
                for _ in range(m):
                    number(0,10**9);number(0,10**9)
            if total>100000:raise ValueError('total_m_exceeds_limit')
        elif key == '005860':
            n=number(1,100000)
            tree_edges(n,((number(1,n)-1,number(1,n)-1) for _ in range(n-1)))
        elif key == '001870':
            for _ in range(number(1,100000)):number(1,100000)
            for _ in range(number(1,100000)):number(1,10000)
        elif key == '008466':
            for _ in range(26):number(-100000,100000)
            s=next(tokens)
            if not 1<=len(s)<=100000 or any(not 'a'<=c<='z' for c in s):
                raise ValueError('invalid_lowercase_string')
        elif key == '009984':
            n=number(1,1000);values=[number(1,10**6) for _ in range(n)]
            if any(a>=b for a,b in zip(values,values[1:])):
                raise ValueError('stations_not_strictly_increasing')
            low=max([Fraction(1)]+[Fraction(v,i) for i,v in enumerate(values,1)])
            high=min(Fraction(v+1,i) for i,v in enumerate(values,1))
            if low>=high:raise ValueError('no_feasible_alpha')
        elif key == '008845':
            n=number(1,2000);m=number(1,2000);q=number(1,200000)
            grid=[next(tokens) for _ in range(n)]
            if any(len(row)!=m or set(row)-{'0','1'} for row in grid):
                raise ValueError('grid_shape_or_alphabet')
            edges=[]
            for i in range(n):
                for j in range(m):
                    if grid[i][j]!='1':continue
                    if i and grid[i-1][j]=='1':edges.append((i*m+j,(i-1)*m+j))
                    if j and grid[i][j-1]=='1':edges.append((i*m+j,i*m+j-1))
            tree_edges(n*m,edges)
            for _ in range(q):
                x=number(1,n);y=number(1,m);number(x,n);number(y,m)
        elif key == '002803':
            n=number(2,300000)
            for _ in range(2):
                if len({number(1,n) for _ in range(n)})!=n:
                    raise ValueError('not_a_permutation')
        elif key == '004138':
            total=0
            for _ in range(number(1,10000)):
                n=number(1,100000);total+=n
                for _ in range(n):number(1,n)
            if total>100000:raise ValueError('total_n_exceeds_limit')
        if next(tokens,None) is not None:raise ValueError('trailing_input_tokens')
    except (ValueError,StopIteration) as exc:
        return str(exc) or 'missing_input_tokens'
    return None
