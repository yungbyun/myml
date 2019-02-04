import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('display.max_columns', 20)

df = pd.read_csv('input/kc_house_data.csv')
#df.describe()
#df.info()
#cols = ['bedrooms', 'sqft_living', 'price']
#print(df[cols].head(10))

def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)


# 첫번째 예측
train_data,test_data = train_test_split(df,train_size = 0.8,random_state=3)

lr = linear_model.LinearRegression()
#print(np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1,1))
X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1,1)

y_train = np.array(train_data['price'], dtype=pd.Series)
#print(y_train)

lr.fit(X_train,y_train)

X_test = np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1,1)
#print(X_test)
y_test = np.array(test_data['price'], dtype=pd.Series)
#print(y_test)

pred = lr.predict(X_test)
msesm = format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f')
rtrsm = format(lr.score(X_train, y_train),'.3f')
rtesm = format(lr.score(X_test, y_test),'.3f')

#print(pred)
#print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))
#print ("Average Price for predicted: {:.3f}".format(pred.mean()))

#print('Intercept: {}'.format(lr.intercept_))
#print('Coefficient: {}'.format(lr.coef_))

evaluation = pd.DataFrame(
    {'Model': [],
    'Details':[],
    'Mean Squared Error (MSE)':[],
    'R-squared (training)':[],
    'Adjusted R-squared (training)':[],
    'R-squared (test)':[],
    'Adjusted R-squared (test)':[]})

r = evaluation.shape[0]
#print(r)
evaluation.loc[r] = ['Simple Model','-',msesm,rtrsm,'-',rtesm,'-']
#print(evaluation)


'''
plt.figure(figsize=(7, 6))
plt.scatter(X_test,y_test,color='darkgreen',label="Data", alpha=.1)
plt.plot(X_test,lr.predict(X_test),color="red",label="Predicted Regression Line")
plt.xlabel("Living Space (sqft)", fontsize=15)
plt.ylabel("Price ($)", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()
'''

sns.set(style="white", font_scale=1)



features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors',
            'waterfront','view','condition','grade','sqft_above','sqft_basement',
            'yr_built','yr_renovated','zipcode','sqft_living15','sqft_lot15']

mask = np.zeros_like(df[features].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="BuGn_r",
            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75});
plt.show()

'''
f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=train_data['bedrooms'],y=train_data['price'], ax=axes[0])
sns.boxplot(x=train_data['floors'],y=train_data['price'], ax=axes[1])
axes[0].set(xlabel='Bedrooms', ylabel='Price')
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Floors', ylabel='Price')

f, axe = plt.subplots(1, 1,figsize=(12.18,5))
sns.boxplot(x=train_data['bathrooms'],y=train_data['price'], ax=axe)
axe.set(xlabel='Bathrooms / Bedrooms', ylabel='Price');
plt.show()


fig=plt.figure(figsize=(19,12.5))
ax=fig.add_subplot(2,2,1, projection="3d")
ax.scatter(train_data['floors'],train_data['bedrooms'],train_data['bathrooms'],c="darkgreen",alpha=.5)
ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nBathrooms / Bedrooms')
ax.set(ylim=[0,12])

ax=fig.add_subplot(2,2,2, projection="3d")
ax.scatter(train_data['floors'],train_data['bedrooms'],train_data['sqft_living'],c="darkgreen",alpha=.5)
ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nsqft Living')
ax.set(ylim=[0,12])

ax=fig.add_subplot(2,2,3, projection="3d")
ax.scatter(train_data['sqft_living'],train_data['sqft_lot'],train_data['bathrooms'],c="darkgreen",alpha=.5)
ax.set(xlabel='\n sqft Living',ylabel='\nsqft Lot',zlabel='\nBathrooms / Bedrooms')
ax.set(ylim=[0,250000])

ax=fig.add_subplot(2,2,4, projection="3d")
ax.scatter(train_data['sqft_living'],train_data['sqft_lot'],train_data['bedrooms'],c="darkgreen",alpha=.5)
ax.set(xlabel='\n sqft Living',ylabel='\nsqft Lot',zlabel='Bedrooms')
ax.set(ylim=[0,250000]);
plt.show()
'''

features1 = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
complex_model_1 = linear_model.LinearRegression()
complex_model_1.fit(train_data[features1],train_data['price'])

#print('Intercept: {}'.format(complex_model_1.intercept_))
#print('Coefficients: {}'.format(complex_model_1.coef_))

pred1 = complex_model_1.predict(test_data[features1])
msecm1 = format(np.sqrt(metrics.mean_squared_error(y_test,pred1)),'.3f')
rtrcm1 = format(complex_model_1.score(train_data[features1],train_data['price']),'.3f')
artrcm1 = format(adjustedR2(complex_model_1.score(train_data[features1],train_data['price']),train_data.shape[0],len(features1)),'.3f')
rtecm1 = format(complex_model_1.score(test_data[features1],test_data['price']),'.3f')
artecm1 = format(adjustedR2(complex_model_1.score(test_data[features1],test_data['price']),test_data.shape[0],len(features1)),'.3f')

r = evaluation.shape[0]
evaluation.loc[r] = ['Complex Model-1','-',msecm1,rtrcm1,artrcm1,rtecm1,artecm1]
#print(evaluation.sort_values(by = 'R-squared (test)', ascending=False))


'''
f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=train_data['waterfront'],y=train_data['price'], ax=axes[0])
sns.boxplot(x=train_data['view'],y=train_data['price'], ax=axes[1])
axes[0].set(xlabel='Waterfront', ylabel='Price')
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='View', ylabel='Price')

f, axe = plt.subplots(1, 1,figsize=(12.18,5))
sns.boxplot(x=train_data['grade'],y=train_data['price'], ax=axe)
axe.set(xlabel='Grade', ylabel='Price');
plt.show()
'''
'''
fig=plt.figure(figsize=(9.5,6.25))
ax=fig.add_subplot(1,1,1, projection="3d")
ax.scatter(train_data['view'],train_data['grade'],train_data['yr_built'],c="darkgreen",alpha=.5)
ax.set(xlabel='\nView',ylabel='\nGrade',zlabel='\nYear Built');
plt.show()
'''


features2 = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
             'grade','yr_built','zipcode']
complex_model_2 = linear_model.LinearRegression()
complex_model_2.fit(train_data[features2],train_data['price'])

#print('Intercept: {}'.format(complex_model_2.intercept_))
#print('Coefficients: {}'.format(complex_model_2.coef_))

pred2 = complex_model_2.predict(test_data[features2])
msecm2 = format(np.sqrt(metrics.mean_squared_error(y_test,pred2)),'.3f')
rtrcm2 = format(complex_model_2.score(train_data[features2],train_data['price']),'.3f')
artrcm2 = format(adjustedR2(complex_model_2.score(train_data[features2],train_data['price']),train_data.shape[0],len(features2)),'.3f')
rtecm2 = format(complex_model_2.score(test_data[features2],test_data['price']),'.3f')
artecm2 = format(adjustedR2(complex_model_2.score(test_data[features2],test_data['price']),test_data.shape[0],len(features2)),'.3f')

r = evaluation.shape[0]
evaluation.loc[r] = ['Complex Model-2','-',msecm2,rtrcm2,artrcm2,rtecm2,artecm2]
#print(evaluation.sort_values(by = 'R-squared (test)', ascending=False))


polyfeat = PolynomialFeatures(degree=2)
X_trainpoly = polyfeat.fit_transform(train_data[features2])
X_testpoly = polyfeat.fit_transform(test_data[features2])
poly = linear_model.LinearRegression().fit(X_trainpoly, train_data['price'])

predp = poly.predict(X_testpoly)
msepoly1 = format(np.sqrt(metrics.mean_squared_error(test_data['price'],pred)),'.3f')
rtrpoly1 = format(poly.score(X_trainpoly,train_data['price']),'.3f')
rtepoly1 = format(poly.score(X_testpoly,test_data['price']),'.3f')

polyfeat = PolynomialFeatures(degree=3)
X_trainpoly = polyfeat.fit_transform(train_data[features2])
X_testpoly = polyfeat.fit_transform(test_data[features2])
poly = linear_model.LinearRegression().fit(X_trainpoly, train_data['price'])

predp = poly.predict(X_testpoly)
msepoly2 = format(np.sqrt(metrics.mean_squared_error(test_data['price'],pred)),'.3f')
rtrpoly2 = format(poly.score(X_trainpoly,train_data['price']),'.3f')
rtepoly2 = format(poly.score(X_testpoly,test_data['price']),'.3f')

r = evaluation.shape[0]
evaluation.loc[r] = ['Polynomial Regression','degree=2',msepoly1,rtrpoly1,'-',rtepoly1,'-']
evaluation.loc[r+1] = ['Polynomial Regression','degree=3',msepoly2,rtrpoly2,'-',rtepoly2,'-']
#print(evaluation.sort_values(by = 'R-squared (test)', ascending=False))

'''
knnreg = KNeighborsRegressor(n_neighbors=15)
knnreg.fit(train_data[features2],train_data['price'])
pred = knnreg.predict(test_data[features2])

mseknn1 = format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f')
rtrknn1 = format(knnreg.score(train_data[features2],train_data['price']),'.3f')
artrknn1 = format(adjustedR2(knnreg.score(train_data[features2],train_data['price']),train_data.shape[0],len(features2)),'.3f')
rteknn1 = format(knnreg.score(test_data[features2],test_data['price']),'.3f')
arteknn1 = format(adjustedR2(knnreg.score(test_data[features2],test_data['price']),test_data.shape[0],len(features2)),'.3f')

knnreg = KNeighborsRegressor(n_neighbors=25)
knnreg.fit(train_data[features2],train_data['price'])
pred = knnreg.predict(test_data[features2])

mseknn2 = format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f')
rtrknn2 = format(knnreg.score(train_data[features2],train_data['price']),'.3f')
artrknn2 = format(adjustedR2(knnreg.score(train_data[features2],train_data['price']),train_data.shape[0],len(features2)),'.3f')
rteknn2 = format(knnreg.score(test_data[features2],test_data['price']),'.3f')
arteknn2 = format(adjustedR2(knnreg.score(test_data[features2],test_data['price']),test_data.shape[0],len(features2)),'.3f')

knnreg = KNeighborsRegressor(n_neighbors=27)
knnreg.fit(train_data[features2],train_data['price'])
pred = knnreg.predict(test_data[features2])

mseknn3 = format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f')
rtrknn3 = format(knnreg.score(train_data[features2],train_data['price']),'.3f')
artrknn3 = format(adjustedR2(knnreg.score(train_data[features2],train_data['price']),train_data.shape[0],len(features2)),'.3f')
rteknn3 = format(knnreg.score(test_data[features2],test_data['price']),'.3f')
arteknn3 = format(adjustedR2(knnreg.score(test_data[features2],test_data['price']),test_data.shape[0],len(features2)),'.3f')

r = evaluation.shape[0]
evaluation.loc[r] = ['KNN Regression','k=15',mseknn1,rtrknn1,artrknn1,rteknn1,arteknn1]
evaluation.loc[r+1] = ['KNN Regression','k=25',mseknn2,rtrknn2,artrknn2,rteknn2,arteknn2]
evaluation.loc[r+2] = ['KNN Regression','k=27',mseknn3,rtrknn3,artrknn3,rteknn3,arteknn3]
print(evaluation.sort_values(by = 'R-squared (test)', ascending=False))
'''


