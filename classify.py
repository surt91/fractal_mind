import numpy as np
from scipy.misc import imread

from convolutional import model

model.load('fractal.tflearn')

filename = "test/good/13.png"
img = imread(filename, mode="RGBA").astype(float)

answer = model.predict([img])

label = ["bad", "good"]
print(label[np.argmax(answer)], answer)
