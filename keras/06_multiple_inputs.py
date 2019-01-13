from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

x_data = [[1., 1], [2, 2], [3, 3], [4, 4]]
y_data = [[1.], [2], [3], [4]]

model = Sequential()
model.add(Dense(1, input_shape=(2, 4)))
model.compile(optimizer=SGD(lr=0.1), loss='mse')
model.fit(x_data, y_data, epochs=1000, batch_size=1)

print(model.get_weights())
print(model.predict([3]))
