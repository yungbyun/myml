# Lab 5 Logistic Regression Classifier
import numpy as np
import tensorflow as tf

x_data = np.array([-2., -1, 1, 2])
y_data = np.array([0., 0, 1, 1])

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
# use sigmoid activation for 0~1 problem
tf.model.add(tf.keras.layers.Activation('sigmoid'))

''' 
better result with loss function == 'binary_crossentropy', try 'mse' for yourself
adding accuracy metric to get accuracy report during training
'''
tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=500)

# Accuracy report
print("Accuracy: ", history.history['acc'][-1])