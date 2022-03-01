import numpy as np
import tensorflow as tf

x = [1]
y = [1]

tf.model = tf.keras.Sequential()
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2

# prints summary of the model to the terminal
#tf.model.summary()

# fit() executes training
history = tf.model.fit(x, y, epochs=100)

# predict() returns predicted value
predict = tf.model.predict(np.array([3, 4, 6, 9]))
print(predict)

# Plot training & validation loss values
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()