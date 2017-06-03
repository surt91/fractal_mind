from convolutional import *

model.load('fractal.tflearn')

X, Y = image_preloader("train", image_shape=(n_side, n_side), mode='folder', grayscale=False, categorical_labels=True)
X_test, Y_test = image_preloader("test", image_shape=(n_side, n_side), mode='folder', grayscale=False, categorical_labels=True)

test_sample(40, X_test, Y_test)
