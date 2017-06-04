"""
This is strongly influenced from the tflearn example:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
import datetime

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import image_preloader

n_side = 128
n_classes = 2

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_90degrees_rotation()
# img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, n_side, n_side, 4],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 4, activation='relu')
network = conv_2d(network, 64, 4, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, n_classes, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

unique = datetime.datetime.now().strftime("%s")
# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=3)


def test_sample(n, X, Y, sample=True):
    from random import random
    import numpy as np
    import matplotlib.pyplot as plt

    label = ["bad", "good", "ugly"]

    if sample:
        for _ in range(n):
            i = int(random() * len(X))
            correct = np.argmax(Y[i])
            predict = np.argmax(model.predict([X[i]]))
            print(label[correct], "<->", label[predict])
            if correct != predict:
                plt.imshow(X[i])
                plt.show()
    else:
        right_positive = 0
        false_positive = 0
        false_negative = 0
        right_negative = 0
        for i in range(len(X)):
            correct = np.argmax(Y[i])
            predict = np.argmax(model.predict([X[i]]))
            if correct == 1 and predict == 1:
                right_positive += 1
            elif correct == 0 and predict == 1:
                false_positive += 1
            elif correct == 1 and predict == 0:
                false_negative += 1
            elif correct == 0 and predict == 0:
                right_negative += 1

            if correct != predict:
                print(label[correct], "<->", label[predict])
                plt.imshow(X[i])
                plt.show()

        print("right_positive", right_positive)
        print("right_negative", right_negative)
        print("false_positive", false_positive)
        print("false_negative", false_negative)


if __name__ == '__main__':
    # Data loading and preprocessing
    # from tflearn.datasets import cifar10
    X, Y = image_preloader("train", image_shape=(n_side, n_side), mode='folder', grayscale=False, categorical_labels=True)
    X_test, Y_test = image_preloader("test", image_shape=(n_side, n_side), mode='folder', grayscale=False, categorical_labels=True)

    model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=64, run_id='fractals_' + unique)

    # save
    model.save('fractal.tflearn')

    test_sample(40, X_test, Y_test, sample=False)
