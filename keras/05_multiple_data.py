import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


x_data = [1,2,3]
y_data = [1,2,3]

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile(optimizer=SGD(lr=0.1), loss='mse')
history = model.fit(x_data, y_data, epochs=10, batch_size=1, verbose=0, callbacks=[PrintDot()])
model.summary()

plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(history.history['loss'])
plt.show();

print(model.get_weights())
print(model.predict([3]))

