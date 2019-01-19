import pandas as pd

# 데이터 로드하기
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 데이터 정보
#print(train.shape)  # 891, 12
#print(test.shape)  # 418, 11
#print(train.head(5))
#print(test.head(5))

# 데이터 체크
#print(train.info())
#print(test.info())

#print(train.isnull().sum())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-dark')
sns.set(font_scale=1.5)

import missingno as msno
import warnings
warnings.filterwarnings('ignore')

aa = (train['Pclass'] == 1) & (train['Survived'] == 1)
#print(train[aa]['Name'])

city_names = pd.Series(['Seoul', 'Busan', 'Jeju'])
#print(city_names)
population = pd.Series([9770000, 3430000, 680000])
print(population)

df1 = pd.DataFrame({'City': city_names, 'Population': population})
print(df1)

# 생존자의 남여 수, 사망자의 남여 수 표시
survived = train[train['Survived'] == 1]['Sex'].value_counts()
dead = train[train['Survived'] == 0]['Sex'].value_counts()
dframe  = pd.DataFrame([survived, dead])
dframe.index = ['Survived', 'Dead']
dframe.plot(kind='bar', stacked=True, figsize=(15, 10))
#plt.show()

