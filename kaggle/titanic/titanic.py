import pandas as pd
import matplotlib.pyplot as plt



import missingno as msno
import warnings
warnings.filterwarnings('ignore')

# ---------------
# 데이터 로드하기
# ---------------
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# ---------------
# 데이터 정보
# ---------------
#print(train.shape)  # 891, 12
#print(test.shape)  # 418, 11
#print(train.head(5))
#print(test.head(5))

# ---------------
# 데이터 체크
# ---------------
#print(train.info())
#print(test.info())

#print(train.isnull().sum())


# ---------------
# 데이터 쿼리
# ---------------
cond = (train['Pclass'] == 1) & (train['Survived'] == 1)
#print(train[cond]['Name'])



# ---------------
# 그래프 표시하기1 (DataFrame)
# ---------------

city_names = pd.Series(['Seoul', 'Busan', 'Jeju'])
#print(city_names)
population = pd.Series([9700000, 3430000, 680000])
#print(population)

df1 = pd.DataFrame({'City': city_names, 'Population': population})
print(df1)



# ---------------
# 그래프 표시하기2
# ---------------

# 생존자의 성별 수, 사망자의 성별 수 표시
sex_servived = train[train['Survived'] == 1]['Sex']
cnt_survived = sex_servived.value_counts()
#print(cnt_survived)
sex_dead = train[train['Survived'] == 0]['Sex']
cnt_dead = sex_dead.value_counts()
#print(cnt_dead)
dframe = pd.DataFrame([cnt_survived, cnt_dead])
dframe.index = ['Survived', 'Dead']
#print(dframe)
#dframe.plot(kind='bar', stacked=True, figsize=(15, 10))
#plt.show()

# 생존자의 좌석등급별 수, 사망자의 좌석등급별 수 표시
sex_servived = train[train['Survived'] == 1]['Pclass']
cnt_survived = sex_servived.value_counts()
sex_dead = train[train['Survived'] == 0]['Pclass']
cnt_dead = sex_dead.value_counts()
dframe = pd.DataFrame([cnt_survived, cnt_dead])
dframe.index = ['Survived', 'Dead']
#dframe.plot(kind='bar', stacked=True, figsize=(15, 10))
#plt.show()

# 형제/배우자 유무별 생존자 수, 형제/배우자 유무별 사망자 수 표시
sex_servived = train[train['Survived'] == 1]['SibSp']
cnt_survived = sex_servived.value_counts()
sex_dead = train[train['Survived'] == 0]['SibSp']
cnt_dead = sex_dead.value_counts()
dframe = pd.DataFrame([cnt_survived, cnt_dead])
dframe.index = ['Survived', 'Dead']
#dframe.plot(kind='bar', stacked=True, figsize=(15, 10))
#plt.show()

# 부모/자식에 따른 생존자 수, 부모/자식에 따른 사망자 수 표시
sex_servived = train[train['Survived'] == 1]['Parch']
cnt_survived = sex_servived.value_counts()
sex_dead = train[train['Survived'] == 0]['Parch']
cnt_dead = sex_dead.value_counts()
dframe = pd.DataFrame([cnt_survived, cnt_dead])
dframe.index = ['Survived', 'Dead']
#dframe.plot(kind='bar', stacked=True, figsize=(15, 10))
#plt.show()

import seaborn as sns

plt.style.use('seaborn-dark')
sns.set(font_scale=1)
selected_column = train[train['Survived'] == 0]['Sex']
count = selected_column.value_counts()
dframe = pd.DataFrame([count])
dframe.index = ['Dead']
#dframe.plot(kind='bar', stacked=False, figsize=(4, 8))
#plt.show()

def xxx(field1, val, field2, x_title):
    plt.style.use('seaborn-dark')
    selected_column = train[train[field1] == val][field2]
    count = selected_column.value_counts()
    dframe = pd.DataFrame([count])
    dframe.index = [x_title]
    dframe.plot(rot=0, kind='bar', title='', stacked=False, figsize=(4, 8), fontsize=12)
    plt.show()

xxx('Parch', 1, 'Embarked', 'Parch: 1')


