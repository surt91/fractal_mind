import os
import sys

import numpy as np
from scipy.misc import imread

from convolutional import model


def download_model():
    import tarfile
    import urllib.request.urlretrieve

    fname = "fractal.tflearn.tar.xz"
    urlretrieve ("https://hendrik.schawe.me/fractal.tflearn.tar.xz", fname)
    with tarfile.open(fname) as f:
        f.extractall('.')
    os.remove(fname)


# TODO: startup is slow, speed it up somehow
if __name__ == '__main__':
    if sys.argc != 2:
        print("this program takes exactly one argument: the filename of the file to be classified")
        sys.exit(1)
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"file {filename} not found")
        sys.exit(2)

    if any(os.path.exists(f) for f in ["fractal.tflearn.data-00000-of-00001", "fractal.tflearn.index", "fractal.tflearn.meta"]):
        download_model()
    model.load('fractal.tflearn')

    img = imread(filename, mode="RGBA").astype(float)

    answer = model.predict([img])

    label = ["bad", "good"]
    print(label[np.argmax(answer)], answer)
