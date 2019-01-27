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
#print(df1)



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

#xxx('Parch', 1, 'Embarked', 'Parch: 1')


# 생존자 수의 구성, 사망자 수의 구성 표시
selected_col_4_servived = train[train['Survived'] == 1]['Sex']
cnt_survived = selected_col_4_servived.value_counts()
selected_col_4_dead = train[train['Survived'] == 0]['Sex']
cnt_dead = selected_col_4_dead.value_counts()
dframe = pd.DataFrame([cnt_survived, cnt_dead])
dframe.index = ['Survived', 'Dead']
#dframe.plot(rot=0, kind='bar', title='Figure for Sex', stacked=True,
#            figsize=(5, 6), fontsize=12)
#plt.show()


def show_sd_fig(col, msg):
    # 생존자 수의 구성, 사망자 수의 구성 표시
    selected_col_for_servived = train[train['Survived'] == 1][col]
    cnt_for_survived = selected_col_for_servived.value_counts()
    selected_col_for_dead = train[train['Survived'] == 0][col]
    cnt_for_dead = selected_col_for_dead.value_counts()
    dataframe = pd.DataFrame([cnt_for_survived, cnt_for_dead])
    dataframe.index = ['Survived', 'Dead']
    dataframe.plot(rot=0, kind='bar', title=msg, stacked=True,
                figsize=(5, 6), fontsize=12)
    plt.show()

#show_sd_fig('SibSp', 'Figure for SibSp')

#print(train.head())

all_data = [train, test]

# 지정한 모든 데이터에서
for record in all_data:
    record['Title'] = record['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

#print(train.head()['Title'])
#print(train['Title'].value_counts())
#print(test['Title'].value_counts())


title_map = {
    'Mr': 0, 'Miss': 1, 'Mrs': 2,
    "Master": 3, "Dr": 3, "Rev": 3,
    "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
    "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3,
    "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in all_data:
    dataset['Title'] = dataset['Title'].map(title_map)

#print(train['Title'].value_counts())
#show_sd_fig('Title', 'Figure for Title')

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

sex_map = {"male": 0, "female": 1}
for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map(sex_map)

#print(test['Sex'].value_counts())

#show_sd_fig('Sex', 'Figure for Sex')

# 호칭별 나이 중간값을 구한 후 비어있는 곳 채우기
#print(train['Age'].head(10))
#print(train['Title'].head(10))

median_train  = train.groupby("Title")["Age"].transform("median")
train["Age"].fillna(median_train , inplace=True)

median_test = test.groupby("Title")["Age"].transform("median")
test["Age"].fillna(median_test, inplace=True)

#print(train['Age'].head(10))

# 나이에 따른 생존자/사망자의 분포
#facet = sns.FacetGrid(train, hue="Survived", aspect=2.5)
#facet.map(sns.kdeplot, 'Age', shade=True)
#facet.set(xlim=(0, train['Age'].max()))
#facet.add_legend()
#plt.show()

# 나이에 따른 탑승장소 분포
#facet = sns.FacetGrid(train, hue="Embarked", aspect=2.5)
#facet.map(sns.kdeplot, 'Age', shade=True)
#facet.set(xlim=(0, train['Age'].max()))
#facet.add_legend()
#plt.xlim(0, 20)
#plt.show()

#train.info()
#test.info()

for dataset in all_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

show_sd_fig('Age', 'Figure for Age')
