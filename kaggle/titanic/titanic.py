import pandas as pd
import matplotlib.pyplot as plt

import missingno as msno
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 20)

# ---------------
# 데이터 로드하기
# ---------------
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
print(test)
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

# 생존자/사망자별 나이 분포
#facet = sns.FacetGrid(train, hue="Survived", aspect=2.5)
#facet.map(sns.kdeplot, 'Age', shade=True)
#facet.set(xlim=(0, train['Age'].max()))
#facet.add_legend()
#plt.show()

# 탑승 장소별 나이 분포
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

#print(train['Age'].head())

#show_sd_fig('Age', 'Figure for Age')

#print(train['Embarked'].value_counts())
#print(train['Embarked'].unique())

#print(train)

# 학습 데이터에서
def show_pclass_fig(col, msg):
    # 티켓 클래스별 ??? 구성 표시
    col1 = train[train['Pclass'] == 1][col]
    cnt1 = col1.value_counts()

    col2 = train[train['Pclass'] == 2][col]
    cnt2 = col2.value_counts()

    col3 = train[train['Pclass'] == 3][col]
    cnt3 = col3.value_counts()

    dataframe = pd.DataFrame([cnt1, cnt2, cnt3])
    dataframe.index = ['1st class', '2nd class', '3rd class']
    dataframe.plot(rot=0, kind='bar', title=msg, stacked=True,
                figsize=(5, 8), fontsize=12)
    plt.show()

#show_pclass_fig('Embarked', 'Figure for Embarked')


#print(train['Embarked'].unique())
#print(train['Embarked'].value_counts())

for dataset in all_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#print(train['Embarked'].unique())
#print(train['Embarked'].value_counts())

#print(test['Fare'])

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in all_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

#print(train['Embarked'].value_counts())

# 좌석 등급(Pclass)이 같은  요금(Fare)들의 중간값으로 빈 요금을 채우라.
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# 비어있는 요금, 좌석 등급(Pclass)이 같은 요금(Fare) 중간값으로
def fill_median(col1, col2):
    train[col1].fillna(train.groupby(col2)[col1].transform("median"), inplace=True)
    test[col1].fillna(test.groupby(col2)[col1].transform("median"), inplace=True)

fill_median('Fare', 'Pclass')


# 탑승 장소별 나이 분포
#facet = sns.FacetGrid(train, hue="Embarked", aspect=2.5)
#facet.map(sns.kdeplot, 'Age', shade=True)
#facet.set(xlim=(0, train['Age'].max()))
#facet.add_legend()
#plt.xlim(0, 20)
#plt.show()

# 생존자/사망자별 요금 분포
#facet = sns.FacetGrid(train, hue="Survived", aspect=4)
#facet.map(sns.kdeplot, 'Fare', shade=True)
#facet.set(xlim=(0, train['Fare'].max()))
#facet.add_legend()
#plt.show()


# 탑승 장소별 요금 분포
#facet = sns.FacetGrid(train, hue="Embarked", aspect=4)
#facet.map(sns.kdeplot, 'Fare', shade=True)
#facet.set(xlim=(0, train['Fare'].max()))
#facet.add_legend()
#plt.xlim(0, 200)
#plt.show()


def show_fare_dist(col):
    facet = sns.FacetGrid(train, hue=col, aspect=4)
    facet.map(sns.kdeplot, 'Fare', shade=True)
    facet.set(xlim=(0, train['Fare'].max()))
    facet.add_legend()
#    plt.xlim(0, 200)
    plt.show()

#show_fare_dist('Embarked')
#show_fare_dist('Sex')
#show_fare_dist('Age')
#show_fare_dist('Pclass')

#print(train['Fare'].head())

# 요금 구간 정보로 바꾸기
for dataset in all_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

#print(train['Fare'].head())

#show_fare_dist('Pclass')


#print(train['Cabin'].value_counts())
#print(train.Cabin)

for dataset in all_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

#print(train['Cabin'].value_counts())

#show_pclass_fig('Cabin', 'Figure for Cabin')


#show_pclass_fig('Fare', 'Figure for Fare')

#print(train['Cabin'].head())

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in all_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

#print(train['Cabin'].head())

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

#print(train['Cabin'].head(10))

#show_pclass_fig('Cabin', 'Figure for Cabin')
#show_sd_fig('Cabin', 'Figure for Cabin')

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
print(test.head(10))

features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
#test = test.drop(['PassengerId'], axis=1)
#print(train.head(10))
#print(test.head(10))

train_input = train.drop('Survived', axis=1)
labels = train['Survived']

#print(train_input.shape, labels.shape)


# ======================================

# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

#print(train.info())

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_input, labels, cv=k_fold, n_jobs=1, scoring=scoring)
#rint(score)
#print(round(np.mean(score)*100, 2))

clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_input, labels, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
#print(round(np.mean(score)*100, 2))

clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_input, labels, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
#print(round(np.mean(score)*100, 2))

clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_input, labels, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
#print(round(np.mean(score)*100, 2))

clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_input, labels, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
#print(round(np.mean(score)*100, 2))


# ========================== Test


# 학습 데이터와 레이블로 학습하고 테스트 데이터로 테스트를 수행하여 결과를 반환함.

def learn_test(clf, train_input, labels, test_input):
    clf.fit(train_input, labels)
    pred = clf.predict(test_input)
    return pred

test_input = test.drop("PassengerId", axis=1).copy()
prediction = learn_test(SVC(), train_input, labels, test_input)

# 제출할 파일을 만든고 이를 화면에 표시함.
df = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

def save_display(df, fname):
    df.to_csv(fname, index=False)

    submission = pd.read_csv(fname)
    print(submission.head())

save_display(df, 'submission.csv')



'''
   PassengerId  Survived
0          892         0
1          893         1
2          894         0
3          895         0
4          896         1
'''
