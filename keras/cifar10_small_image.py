
#Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


