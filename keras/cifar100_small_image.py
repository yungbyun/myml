
# https://keras.io/datasets/
# Dataset of 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images.

from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')



