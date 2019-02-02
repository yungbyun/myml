import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import sklearn

plt.style.use('seaborn')
sns.set(font_scale=1.2) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.
import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore') # 워닝 메세지를 생략해 줍니다. 차후 버전관리를 위해 필요한 정보라고 생각하시면 주석처리 하시면 됩니다.

#print(os.listdir("./data"))


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')
#print(train.shape)
#print(test.shape)
#print(sample_submission.shape)
#print(sample_submission.head())


#print(train.columns)
#print(test.columns)
#print(sample_submission.columns)

pd.set_option('display.max_columns', 20)
#print(train.head())
#print(test.head())
#print(train.dtypes)

#print(train.describe())
#print(test.describe())
#print(sample_submission.describe())

#print(train.isnull().sum() / train.shape[0])
#print(test.isnull().sum() / train.shape[0])


'''

f, ax = plt.subplots(1, 1, figsize=(6, 5.5))
train['Sex'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, shadow=True)
ax.set_title('Pie Chart for Sex')
ax.set_ylabel('')
plt.show()



f, ax = plt.subplots(1, 1, figsize=(4, 7))
sns.countplot('Embarked', data=train, ax=ax)
ax.set_title('Bar Chart for Embarked')
plt.show()
'''

print(train.columns)
tmp = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
#print(tmp)

#print(train.Survived.head())
tmp = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
#print(tmp)

tmp = pd.crosstab(train['Pclass'], train['Survived'], margins=True)
#print(tmp)

tmp = pd.crosstab(train['Sex'], train['Survived'], margins=True)
#print(tmp)

tmp = pd.crosstab(train['Survived'], train['Embarked'], margins=True)
#print(tmp)

tmp = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()
#print(tmp)

'''
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
plt.show()
'''

'''
train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar()
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()
'''


'''
train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar()
sns.countplot('Sex', hue='Survived', data=train)
plt.show()


# 좌석 클래스에 따른 남성 생존률, 좌석 클래스에 따른 여성 생존률
sns.factorplot('Pclass', 'Survived', hue='Embarked', data=train,
    size=6, aspect=1.5)
plt.show()
'''

#print('Oldest : {:.1f} Years'.format(train['Age'].max()))
#print('Youngest : {:.1f} Years'.format(train['Age'].min()))
#print('Average : {:.1f} Years'.format(train['Age'].mean()))


'''

plt.figure(figsize=(8, 6))
train['Age'][train['Pclass'] == 1].plot(kind='kde')
train['Age'][train['Pclass'] == 2].plot(kind='kde')
train['Age'][train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Distribution of age')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
plt.show()


print(train['Embarked'].value_counts())

plt.figure(figsize=(8, 6))
train['Age'][train['Embarked'] == 'S'].plot(kind='kde')
train['Age'][train['Embarked'] == 'C'].plot(kind='kde')
train['Age'][train['Embarked'] == 'Q'].plot(kind='kde')

plt.xlabel('Age')
plt.title('Distribution of age')
plt.legend(['S', 'C', 'Q'])
plt.show()


ratio_list=[]
for i in range(1, 80):
    # 0, 1구분없이 갯수 (1살 이하 모든 사람 수)
    total = len(train[train['Age'] < i]['Survived'])
    # 1만 더함 (1살 이하 산 사람 수)
    servived = train[train['Age'] < i]['Survived'].sum()
    ratio = servived / total
    ratio_list.append(ratio)

plt.figure(figsize=(7, 7))
plt.plot(ratio_list)
plt.title('Survival Ratio', y=1.02)
plt.ylabel('Ratio')
plt.xlabel('Age')
plt.show()


print(train['Embarked'].unique())

#탑승 지역에 따른  생존 확률
f, ax = plt.subplots(1, 1, figsize=(5, 7))
groupby = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True)
groupby.mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
plt.show()


f,ax=plt.subplots(1, 1, figsize=(7,6))
sns.countplot('Embarked', data=train, ax=ax)
ax.set_title('Number of Passengers')
plt.show()


f,ax=plt.subplots(1, 1, figsize=(7,6))
sns.countplot('Embarked', hue='Sex', data=train, ax=ax)
ax.set_title('Number of passengers')
plt.show()


f,ax=plt.subplots(1, 1, figsize=(7,6))
sns.countplot('Embarked', hue='Survived', data=train, ax=ax)
ax.set_title('Embarked vs Survived')
plt.show()



f,ax=plt.subplots(1, 1, figsize=(7,6))
sns.countplot('Embarked', hue='Pclass', data=train, ax=ax)
ax.set_title('Seats for Embarked')
plt.show()
'''

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

'''
f,ax=plt.subplots(1, 1, figsize=(8,7))
sns.countplot('FamilySize', data=train, ax=ax)
ax.set_title('Number of passengers boarded')
plt.show()


f,ax=plt.subplots(1, 1, figsize=(8,7))
sns.countplot('FamilySize', hue='Survived', data=train, ax=ax)
ax.set_title('Number of passengers boarded')
plt.show()


f,ax=plt.subplots(1, 1, figsize=(8,7))
a=train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True)
print(a)
a.mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
ax.set_title('Survival ratio according to FamilySize')
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(train['Fare'], color='b', ax=ax)
g = g.legend(loc='best')
'''

# 특이하기도 train set 말고 test set에 Fare 피쳐에 널 값이 하나 존재하는 것을 확인할 수 있었습니다.
# 그래서 평균 값으로 해당 널값을 넣어줍니다.
m=test['Fare'].mean()
#print(m)
# testset 에 있는 nan value를 평균값으로 치환합니다.
#print(test.Fare.isnull())
# test 데이터의 Fare 컬럼에서 null인곳만 m 할당
test.loc[test.Fare.isnull(), 'Fare'] = m

# Fare를 조금 마사지
train['Fare'] = train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
test['Fare'] = test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

'''
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(train['Fare'], color='b', ax=ax)
g = g.legend(loc='best')
plt.show()
'''

#print(train["Cabin"].isnull().sum() / train.shape[0])

#print(train['Ticket'].value_counts())


# Feature Engineering

#print(train["Age"].isnull().sum())

# 호칭 추출하기
train['Title']= train.Name.str.extract('([A-Za-z]+)\.')
test['Title']= test.Name.str.extract('([A-Za-z]+)\.')

train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

'''
print(train.groupby('Title').mean())
train.groupby('Title')['Survived'].mean().plot.bar()
plt.show()
'''
#print(train.groupby('Title').mean())

#에고 직접 값을
train.loc[(train.Age.isnull())&(train.Title=='Mr'),'Age'] = 33
train.loc[(train.Age.isnull())&(train.Title=='Mrs'),'Age'] = 36
train.loc[(train.Age.isnull())&(train.Title=='Master'),'Age'] = 5
train.loc[(train.Age.isnull())&(train.Title=='Miss'),'Age'] = 22
train.loc[(train.Age.isnull())&(train.Title=='Other'),'Age'] = 46

test.loc[(test.Age.isnull())&(test.Title=='Mr'),'Age'] = 33
test.loc[(test.Age.isnull())&(test.Title=='Mrs'),'Age'] = 36
test.loc[(test.Age.isnull())&(test.Title=='Master'),'Age'] = 5
test.loc[(test.Age.isnull())&(test.Title=='Miss'),'Age'] = 22
test.loc[(test.Age.isnull())&(test.Title=='Other'),'Age'] = 46

#print(train.isnull().sum()[train.isnull().sum() > 0])
#print(test.isnull().sum()[test.isnull().sum() > 0])

#print(train['Embarked'].isnull().sum())


train['Embarked'].fillna('S', inplace=True)

#print(train.isnull().sum()[train.isnull().sum() > 0])


def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7

train['Age_cat'] = train['Age'].apply(category_age)
test['Age_cat'] = test['Age'].apply(category_age)

#print(train.groupby(['Age_cat'])['PassengerId'].count())

train['Title'] = train['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
test['Title'] = test['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test['Embarked'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

#print(train['Embarked'].isnull().any(), train['Embarked'].dtypes)

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})


'''
heatmap_data = train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Title', 'Age_cat', 'Age']]
colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})
plt.show()
'''


train = pd.get_dummies(train, columns=['Title'], prefix='Title')
test = pd.get_dummies(test, columns=['Title'], prefix='Title')


#print(train.head())
train = pd.get_dummies(train, columns=['Embarked'], prefix='Embarked')
test = pd.get_dummies(test, columns=['Embarked'], prefix='Embarked')
#print(train.head())

train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

#print(train.head())
#print(test.head())

#print(train.dtypes)
#print(test.dtypes)


#=================================

#importing all the required ML packages
# 유명한 randomforestclassfier 입니다.
from sklearn.ensemble import RandomForestClassifier
# 모델의 평가를 위해서 씁니다
from sklearn import metrics
# traning set을 쉽게 나눠주는 함수입니다.
from sklearn.model_selection import train_test_split


X_train = train.drop('Survived', axis=1).values
target_label = train['Survived'].values
X_test = test.values


X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.2, random_state=2018)
#print(y_tr.shape, y_vld.shape)

#============= 모델 생성 및 학습

'''
model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)
#print(prediction)


#print('인식률 : {:.2f}%'.format(
#    100 * metrics.accuracy_score(prediction, y_vld)))


from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index=test.columns)
'''

'''
plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam, SGD


nn_model = Sequential()
nn_model.add(Dense(32,activation='relu',input_shape=(14,)))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(64,activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(32,activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1,activation='sigmoid'))

Loss = 'binary_crossentropy'
nn_model.compile(loss=Loss,optimizer=Adam(),metrics=['accuracy'])
nn_model.summary()

#print(X_train.shape, X_test.shape)
history = nn_model.fit(X_tr,y_tr,
    batch_size=64,
    epochs=500,
    validation_data=(X_vld, y_vld),
    verbose=0)


history_list = [history] # to list
#print(history_list)

hist_df = pd.concat([pd.DataFrame(history_item.history) for history_item in history_list], sort=True)
#print(hist_df.head())

hist_df.index = np.arange(1, len(hist_df)+1)
#print(hist_df.head(20))


'''
fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(15, 7))
axs.plot(hist_df.val_acc, lw=2, label='Validation')
axs.plot(hist_df.acc, lw=2, label='Training')
axs.set_ylabel('Accuracy')
axs.set_xlabel('Epoch')
axs.grid()
axs.legend(loc=0)
fig.savefig('hist.png', dpi=300)
plt.show()


fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(15, 7))
axs.plot(hist_df.val_loss, lw=2, label='Validation')
axs.plot(hist_df.loss, lw=2, label='Training')
axs.set_ylabel('Loss')
axs.set_xlabel('Epoch')
axs.grid()
axs.legend(loc=0)
fig.savefig('hist.png', dpi=300)
plt.show()
'''

submission = pd.read_csv('data/sample_submission.csv')
print(submission.head(10))

prediction = nn_model.predict(X_test)
prediction = prediction > 0.5
prediction = prediction.astype(np.int)
prediction = prediction.T[0]
print(prediction)
print(prediction.shape)


submission['Survived'] = prediction
print(submission.head(10))
submission.to_csv('my_nn_submission.csv', index=False)


