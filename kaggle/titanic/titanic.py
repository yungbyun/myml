import pandas as pd
import numpy as np
np.core

# 데이터 로드하기
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 데이터 정보
#print(train.shape)  # 891, 12
#print(test.shape)  # 418, 11
print(train.head(5))
#print(test.head(5))

# 데이터 체크
#print(train.info())
#print(test.info())

#print(train.isnull().sum())

import matplotlib.pyplot as plt

# 성별 생존자와 사망자 표시하기
survived = train[train['Survived'] == 1]['Sex'].value_counts()
dead = train[train['Survived'] == 0]['Sex'].value_counts()
df = pd.DataFrame([survived, dead])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(10, 5))
plt.show()


