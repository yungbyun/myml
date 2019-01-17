import pandas as pd
print(pd.__version__)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
print(city_names)
population = pd.Series([852469, 1015785, 485199])

df1 = pd.DataFrame({'City name': city_names, 'Population': population})
print(df1)

housing_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
#print(housing_df.describe())
print(housing_df.head())

cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(type(cities['City name']))
cities['City name']

