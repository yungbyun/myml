import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', 20)

# 그래프에서 격자로 숫자 범위가 눈에 잘 띄도록 ggplot 스타일을 사용
plt.style.use('ggplot')

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

train = pd.read_csv("data/train.csv", parse_dates=["datetime"])
#print(train.shape)

#train.info()

#print(train.head())

#print(train.temp.describe())

#print(train.isnull().sum())

import missingno as msno
#msno.matrix(train, figsize=(12,7), fontsize=10)
#plt.show()

#print(train.shape)
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second
#print(train.shape)

#print(train.head())

'''
figure, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(12,12)

sns.barplot(data=train, x="year", y="count", ax=ax1)
sns.barplot(data=train, x="month", y="count", ax=ax2)
sns.barplot(data=train, x="day", y="count", ax=ax3)
sns.barplot(data=train, x="hour", y="count", ax=ax4)

ax1.set(ylabel='Count',title="Amount of rent(yearly)")
ax2.set(xlabel='month',title="Amount of rent(monthly)")
ax3.set(xlabel='day', title="Amount of rent(daily)")
ax4.set(xlabel='hour', title="Amount of rent(hourly)")

plt.show()


fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 13)
sns.boxplot(data=train,y="count",orient="v",ax=axes[0][0])
sns.boxplot(data=train,y="count",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=train,y="count",x="hour",orient="v",ax=axes[1][0])
sns.boxplot(data=train,y="count",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Amount of rent")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Rent(yearly)")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Rent(houly)")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Rent(Working/No working)")
plt.show()
'''

#print(train.shape)
train["dayofweek"] = train["datetime"].dt.dayofweek
#print(train.shape)


#print(train["dayofweek"].value_counts())

'''
fig,(ax1,ax2)= plt.subplots(nrows=2)
fig.set_size_inches(12,7)
sns.pointplot(fontsize=10, data=train, x="hour", y="count", hue="workingday", ax=ax1)
sns.pointplot(fontsize=10, data=train, x="hour", y="count", hue="dayofweek", ax=ax2)
plt.show()

fig,(ax1,ax2)= plt.subplots(nrows=2)
fig.set_size_inches(12,7)
sns.pointplot(data=train, x="hour", y="count", hue="weather", ax=ax1)
sns.pointplot(data=train, x="hour", y="count", hue="season", ax=ax2)
plt.show()


corrMatt = train[["temp", "atemp", "casual", "registered",
                  "humidity", "windspeed", "count"]]
corrMatt = corrMatt.corr()
#print(corrMatt)

mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
plt.show()


fig, ax1 = plt.subplots(ncols=1)
fig.set_size_inches(7, 5)
sns.regplot(x="temp", y="count", data=train,ax=ax1)

fig, ax2 = plt.subplots(ncols=1)
fig.set_size_inches(7, 5)
sns.regplot(x="windspeed", y="count", data=train,ax=ax2)

fig, ax3 = plt.subplots(ncols=1)
fig.set_size_inches(7, 5)
sns.regplot(x="humidity", y="count", data=train,ax=ax3)
plt.show()
'''

def concatenate_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)

train["year_month"] = train["datetime"].apply(concatenate_year_month)

#print(train.shape)
#print(train[["datetime", "year_month"]].head())


'''
fig, ax1 = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(2, 3)
sns.barplot(data=train, x="year", y="count", ax=ax1)
plt.show()

fig, ax2 = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(8, 4)
sns.barplot(data=train, x="month", y="count", ax=ax2)
plt.show()


fig, ax3 = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(12, 7)
sns.barplot(data=train, x="year_month", y="count", ax=ax3)
plt.show()
'''

# trainWithoutOutliers
trainWithoutOutliers = \
    train[np.abs(train["count"] - train["count"].mean()) <= (3*train["count"].std())]

print(train.shape)
print(trainWithoutOutliers.shape)


'''
# count값의 데이터 분포도를 파악
figure, axes = plt.subplots(ncols=2, nrows=2)
figure.set_size_inches(12, 10)
sns.distplot(train["count"], ax=axes[0][0])
stats.probplot(train["count"], dist='norm', fit=True, plot=axes[0][1])
sns.distplot(np.log(trainWithoutOutliers["count"]), ax=axes[1][0])
stats.probplot(np.log1p(trainWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])
plt.show()
'''
