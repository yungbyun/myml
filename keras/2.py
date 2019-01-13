import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy as np
from numpy.random import random

x = random((30, 2))
w = np.array([3., 2.])
b = 1.
y = np.dot(x, w) + b

model = Sequential()
# 30 observations. Each observation has 2 features.
model.add(Dense(1, input_shape=(2,)))
# MSE because we want linear regression.
model.compile(optimizer=SGD(lr=0.1), loss='mse')
model.fit(x, y, nb_epoch=60, batch_size=1)

print(model.get_weights())
