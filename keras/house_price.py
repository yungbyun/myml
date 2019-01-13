
# https://www.tensorflow.org/tutorials/keras/basic_regression

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('house_price.csv', delim_whitespace=True, header=None)

data_set = df.values
X = data_set[:, 0:13]
print(X)
Y = data_set[:, 13]

X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train, epochs=200, batch_size=10)

Y_prediction = model.predict(X_validation).flatten()

for i in range(10):
    real_price = Y_validation[i]
    predicted_price = Y_prediction[i]
    print('Real Price: {:.3f}, Predicted Price: {:.3f}'.format(real_price, predicted_price))

model.summary()