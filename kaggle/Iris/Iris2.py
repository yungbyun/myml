
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', 20)

iris = pd.read_csv("data/Iris.csv") #load the dataset
print(iris.head())


print(iris.info())  #checking if there is any inconsistency in the dataset
#as we see there are no null values in the dataset, so the data can be processed



iris.drop('Id',axis=1,inplace=True)
print(iris.head())
#dropping the Id column as it is unecessary, axis=1 specifies that it should be column wise, inplace =1 means the changes should be reflected into the dataframe


