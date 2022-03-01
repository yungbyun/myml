# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np

x_data = np.array([[73., 80.],
          [93., 88.],
          [89., 91.],
          [73., 66.]])
y_data = np.array([[152.],
          [185.],
          [180.],
          [142.]])

tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))  # input_dim=3 gives multi-variable regression
tf.model.add(tf.keras.layers.Activation('linear'))  # this line can be omitted, as linear activation is default
# advanced reading https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
#tf.model.summary()
history = tf.model.fit(x_data, y_data, epochs=100)

y_predict = tf.model.predict(np.array([[72., 93.]]))
print(y_predict)

# Plot training & validation loss values
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()