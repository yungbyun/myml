
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')
#print(df)

'''
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.show()
'''

area = df[['area']]
#print(area)

price = df.price
#print(price)

# Create linear modelression object
model = linear_model.LinearRegression()
model.fit(area,price)

predicted = model.predict([[3000]])
#print(predicted)

# Y = m * X + b (m is coefficient and b is intercept)
#print(model.coef_)
#print(model.intercept_)


# Generate CSV file with list of home price predictions
area_df = pd.read_csv("areas.csv")
#print(area_df.head(3))

p = model.predict(area_df)
#print(p)

area_df['prices']=p
#print(area_df)

area_df.to_csv("prediction.csv")


plt.xlabel('area')
plt.ylabel('price', fontsize=10)
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,model.predict(area),color='blue')
plt.show()
