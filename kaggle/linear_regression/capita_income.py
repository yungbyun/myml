
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('canada_per_capita_income.csv')
#print(df.head(10))
print(df.info())
'''
plt.xlabel('Year')
plt.ylabel('Capita Income')
plt.scatter(df.year,df.income,color='red',marker='+')
plt.show()
'''
year = df[['year']]
print(year.head(10))

income = df.income
print(income.head(10))

# Create linear modelression object
model = linear_model.LinearRegression()
model.fit(year,income)

predicted = model.predict([[2020]])
print(predicted)

# Y = m * X + b (m is coefficient and b is intercept)
print(model.coef_)
print(model.intercept_)

# Generate CSV file with list of home price predictions
year_df = pd.read_csv("year.csv")
print(year_df.head(5))

p = model.predict(year_df)
print(p)

year_df['income']=p
print(year_df)

year_df.to_csv("predicted_income.csv")
'''
plt.xlabel('Year')
plt.ylabel('Income', fontsize=10)
plt.scatter(year,income,color='red',marker='+')
plt.plot(df['year'],model.predict(year),color='blue')
plt.show()
'''
