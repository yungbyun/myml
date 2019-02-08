import numpy as np
import pandas as pd

s = pd.Series([1, 2, 3, np.nan, 5, 6])
#print(s)

dates = pd.date_range('20190101', periods=6)
#print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
#print(df)

df2 = pd.DataFrame({'A': 1.,
	'B': pd.Timestamp('20190101'),
	'C': pd.Series(1, index=list(range(4)), dtype='float32'),
	'D': np.array([3] * 4, dtype='int32'),
	'E': pd.Categorical(["test", "train", "test", "train"]),
	'F': 'foo'})
#print(df2)
#print(df2.dtypes)

#print(df.tail(3))
#print(df.index)
#print(df.columns)
#print(df.describe())

#print(df)
pd.set_option('display.max_columns', 20)
#print(df.T)


#print(df.sort_index(axis=1, ascending=False))


#print(df.loc[dates[0]])
#print(df.loc[:, ['A', 'B']])
#print(df.loc['20190103':'20190105', ['A', 'B']])
#print(df.loc['20190103', ['A', 'B']])
#print(df.loc[dates[0], 'A'])
#print(df.iloc[3])
#print(df.iloc[3:5, 0:2])
#print(df.iloc[[1, 2, 4], [0, 2]])
#print(df.iloc[1:3, :])
#print(df.iloc[:, 1:3])
#print(df)
#print(df.iloc[1, 1])
#print(df[df > 0])

df2 = df.copy()
#print(df2)
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']

#print(df2)
#print(df2['E'].isin(['two', 'four']))
#print(df2[df2['E'].isin(['two', 'four'])])

#print(df)
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20190101', periods=6))
#print(s1)
df['F'] = s1
#print(df)

df.at[dates[0], 'A'] = 0
#print(df)

df.iat[0, 1] = 0

#print(np.array([5] * 6))

#print(df)
df.loc[:, 'D'] = np.array([3] * len(df))
#print(df)

df2 = df.copy()
#print(df2)
df2[df2 > 0] = -df2
#print(df2)

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
#print(df1)

df1.loc[dates[0]:dates[1], 'E'] = 1
a = df1.dropna(how='any')
a = pd.isna(df1)
#print(a)

a = df.mean(1) # axis=1
a = df.mean(0) # axis=0

a = pd.Series([1, 1, 1, np.nan, 1, 1], index=dates)

#print(df)
a = pd.Series([1, 1, 1, np.nan, 1, 1], index=dates).shift(2)
#print(a)
a = df.sub(a, axis='index')
#print(a)


#print(df)
a = df.apply(np.cumsum)
a = df.apply(lambda x: x.max() - x.min())
#print(a)


s = pd.Series(np.random.randint(0, 7, size=10))
#print(s)
#print(s.value_counts())


s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
a = s.str.lower()

df = pd.DataFrame(np.random.randn(10, 4))
#print(df)

piece = df[:3]
#print(piece)
piece = df[3:7]
#print(piece)
piece = df[7:]
#print(piece)

pieces = [df[:3], df[3:7], df[7:]]
p=pd.concat(pieces)
#print(p)

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
#print(left)
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
#print(right)
a=pd.merge(left, right, on='key')
#print(a)

left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
#print(left)
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
#print(right)
a=pd.merge(left, right, on='key')
#print(a)

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
#print(df)

s = df.iloc[3]
#print(s)
a = df.append(s, ignore_index=True)
#print(a)




df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
    'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'three',
    'two', 'two', 'one', 'three'],
    'C': np.random.randn(8),
    'D': np.random.randn(8)})

#print(df)

a = df.groupby('A').sum()
#print(a)

a = df.groupby(['A', 'B']).sum()
#print(a)


tuples = list(zip(*[
    ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))

#print(tuples)

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]

#print(df2)
stacked = df2.stack()
#print(stacked.unstack())

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
   'B': ['A', 'B', 'C'] * 4,
   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
   'D': np.random.randn(12),
   'E': np.random.randn(12)})
#print(df)

pt = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
#print(pt)

#간격을 초로
rng = pd.date_range('20190101', periods=100, freq='S')
#print(rng)

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
#print(ts)
rs = ts.resample('5Min').sum()

#간격을 날짜로
rng = pd.date_range('1/1/2019 00:00', periods=10, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
#print(ts)

rng = pd.date_range('1/1/2019 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)

ts_utc = ts.tz_localize('UTC')

a = ts_utc.tz_convert('US/Eastern')
#print(a)

rng = pd.date_range('20190101', periods=5, freq='M')
#print(rng)


ts = pd.Series(np.random.randn(len(rng)), index=rng)
#print(ts)

ps = ts.to_period()
#print(ps)
a = ps.to_timestamp()
#print(a)


df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
    "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
#print(df)

a = df["grade"].cat.categories = ["very good", "good", "very bad"]
#print(a)

a = df.sort_values(by="grade")
#print(a)

a = df.groupby("grade").size()
#print(a)

import matplotlib.pyplot as plt

ts = pd.Series(
    np.random.randn(1000),
    index=pd.date_range('1/1/2019', periods=1000))
ts = ts.cumsum()
ts.plot(figsize=(15, 5))
#plt.show()

df = pd.DataFrame(
    np.random.randn(1000, 4), index=ts.index,
    columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
df.plot(figsize=(15, 5))
plt.legend(loc='best')
plt.show()











'''

df.to_csv('foo.csv')
a = pd.read_csv('foo.csv')
#print(a)

df.to_hdf('foo.h5', 'df')
a = pd.read_hdf('foo.h5', 'df')

df.to_excel('foo.xlsx', sheet_name='Sheet1')
a = pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
#print(a)
'''