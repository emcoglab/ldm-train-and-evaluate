from pandas import DataFrame

df = DataFrame({'val':  [ 2,    3,    5,    7  ],
                'foo':  ['f1', 'f2', 'f3', 'f4']},
               index=['n1', 'n2', 'n3', 'n4'])

v = df.val.values[:, None] * df.val.values

x = df.foo.values[:, None] + df.foo.values

d = DataFrame(x, df.index, df.index)

pass
