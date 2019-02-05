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

ratio_list=[]
for i in range(1, 80):
    # 0, 1 구분없이 갯수(i살 이하 모든 사람 수)
    total = len(train[train['Age'] < i]['Survived'])# 1인 것들(생존자들)을 모두 더함(1살 이하 산 사람 수)
    servived = train[train['Age'] < i]['Survived'].sum()
    ratio = servived / total
    ratio_list.append(ratio)

print(ratio_list)

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
#print(train.head(10))

train['Initial']= train.Name.str.extract('([A-Za-z]+)\.')
test['Initial']= test.Name.str.extract('([A-Za-z]+)\.')

print(train['Initial'].head(10))
train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'], ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'], inplace=True)
print(train['Initial'].head(10))

