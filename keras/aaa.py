import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

X = np.arange(-10, 11, 1)
Y = X * X + 4

print(X)
print(Y)

model = keras.models.Sequential()
model.add(keras.layers.Dense(10, input_shape=(1,))) #1개씩 여러번
model.add(keras.layers.Dense(20))
model.add(keras.layers.Dense(1))
model.compile(loss="mse", optimizer='sgd')
model.fit(X, Y, verbose=0, epochs=50000)
predicted = model.predict(X)

print(predicted)
print(Y)

import matplotlib.pyplot as plt

plt.plot(X, Y)
plt.plot(X, predicted)
plt.show()



