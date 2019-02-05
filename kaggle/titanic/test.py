import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 20)

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
#print(train['Cabin'].head())
#print(test.isnull().sum() / test.shape[0])

# 생존자/사망자별 승선지 정보 표시
tmp = pd.crosstab(train['Survived'], train['Embarked'], margins=True)
#print(tmp)

tmp = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()
#print(tmp)

#train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
#plt.show()

#sns.countplot('Pclass', hue='Survived', data=train)
#plt.show()

#sns.countplot('Sex', hue='Survived', data=train)
#plt.show()

#sns.factorplot('Embarked', 'Survived', hue='Sex', data=train,size=6, aspect=1.5)
#plt.show()


'''
plt.figure(figsize=(8, 6))
train['Age'][train['Embarked'] == 'S'].plot(kind='kde')
train['Age'][train['Embarked'] == 'C'].plot(kind='kde')
train['Age'][train['Embarked'] == 'Q'].plot(kind='kde')
plt.xlabel('Age')
plt.title('Distribution of age')
plt.legend(['S', 'C', 'Q'])
plt.show()
'''

m=test['Fare'].mean()
test.loc[test.Fare.isnull(), 'Fare'] = m

'''
ratio_list=[]
for i in range(1, 80):
    # 0, 1 구분없이 갯수(i살 이하 모든 사람 수)
    total = len(train[train['Age'] < i]['Survived'])# 1인 것들(생존자들)을 모두 더함(1살 이하 산 사람 수)
    servived = train[train['Age'] < i]['Survived'].sum()
    ratio = servived / total
    ratio_list.append(ratio)

#print(ratio_list)
'''

'''
plt.figure(figsize=(7, 7))
plt.plot(ratio_list)

plt.title('Survival Rate', y=1.02)
plt.ylabel('Survival Rate')
plt.xlabel('Age')
plt.show()
'''

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

train['Title']= train.Name.str.extract('([A-Za-z]+)\.')
test['Title']= test.Name.str.extract('([A-Za-z]+)\.')

train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'], ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'], inplace=True)

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

train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'], inplace=True)

train['Title'] = train['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
test['Title'] = test['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test['Embarked'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

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

train['Age_category'] = train['Age'].apply(category_age)
test['Age_category'] = test['Age'].apply(category_age)

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})

train = pd.get_dummies(train, columns=['Title'], prefix='Title')
test = pd.get_dummies(test, columns=['Title'], prefix='Title')

train = pd.get_dummies(train, columns=['Embarked'], prefix='Embarked')
test = pd.get_dummies(test, columns=['Embarked'], prefix='Embarked')

train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)


#------------- 데이터 준비

# 랜덤 포레스트 분류기 알고리즘
from sklearn.ensemble import RandomForestClassifier
# 모델 평가
from sklearn import metrics
# 데이터 쪼개기
from sklearn.model_selection import train_test_split

X_train = train.drop('Survived', axis=1).values # copy
target_label = train['Survived'].values # copy

X_test = test.values
print(X_train.shape, X_test.shape) # (891, 14) (418, 14)

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.2, random_state=2018)
print(y_tr.shape, y_vld.shape)


#------------- 모델 생성 및 실험

model = RandomForestClassifier()
model.fit(X_tr, y_tr)

prediction = model.predict(X_vld)

print('인식률: {:.2f}%'.format(100 * metrics.accuracy_score (prediction, y_vld)))

feature_importance = model.feature_importances_
Series_feat_imp = pd.Series(feature_importance, index=test.columns)

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
nn_model.add(Dense(32,activation='relu', input_shape=(14,)))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(64,activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(32,activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1,activation='sigmoid'))

Loss = 'binary_crossentropy'
nn_model.compile(loss=Loss,optimizer=Adam(),metrics=['accuracy'])
nn_model.summary()

print('Training...')
history = nn_model.fit(X_tr,y_tr,
    batch_size=64,
    epochs=500,
    validation_data=(X_vld, y_vld),
    verbose=0)

history_list = [history] # to list
hist_df = pd.concat([pd.DataFrame(history_item.history) for
	history_item in history_list], sort=True)
print(hist_df.head())

import numpy as np
hist_df.index = np.arange(1, len(hist_df)+1)

'''
fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(12, 4))
axs.plot(hist_df.val_acc, lw=1, label='Validation')
axs.plot(hist_df.acc, lw=1, label='Training')
axs.set_ylabel('Accuracy')
axs.set_xlabel('Epoch')
axs.grid()
axs.legend(loc=0)
fig.savefig('hist.png', dpi=300)
plt.show()
'''
'''
fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(15, 7))
axs.plot(hist_df.val_loss, lw=1, label='Validation')
axs.plot(hist_df.loss, lw=1, label='Training')
axs.set_ylabel('Loss')
axs.set_xlabel('Epoch')
axs.grid()
axs.legend(loc=0)
fig.savefig('hist.png', dpi=300)
plt.show()
'''

submission = pd.read_csv('input/gender_submission.csv')
print(submission.head(10))

prediction = nn_model.predict(X_test)
prediction = prediction > 0.5
prediction = prediction.astype(np.int)
prediction = prediction.T[0]
print(prediction)
print(prediction.shape)

submission['Survived'] = prediction
print(submission.head(10))
submission.to_csv('input/my_nn_submission.csv', index=False)






