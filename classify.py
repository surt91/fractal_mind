import os
import sys
import shutil

import numpy as np
from scipy.misc import imread, imresize, imsave

from convolutional import model

prog_path = os.path.dirname(os.path.realpath(__file__))


def download_model():
    print("starting download of model")
    import tarfile
    import urllib.request

    fname = "fractal.tflearn.tar.xz"
    urllib.request.urlretrieve("https://hendrik.schawe.me/fractal.tflearn.tar.xz", fname)

    print("unpack model")
    with tarfile.open(fname) as f:
        f.extractall('.')
    os.remove(fname)


class Classifier:
    def __init__(self):
        # download model if necessary
        missing_files = [not os.path.exists(os.path.join(prog_path, f)) for f in ["fractal.tflearn.data-00000-of-00001", "fractal.tflearn.index", "fractal.tflearn.meta"]]
        if any(missing_files):
            download_model()
        model.load(os.path.join(prog_path, 'fractal.tflearn'))

    def analyse(self, images):
        answers = []
        for filename in images:
            if not os.path.exists(filename):
                print(f"file {filename} not found")
                sys.exit(2)

            img = imread(filename, mode="RGBA")
            # crop
            y, x, _ = img.shape
            if x < y:
                img = img[y//2-x//2:y//2+x//2,:]
            elif x > y:
                img = img[:,x//2-y//2:x//2+y//2]
            # resize
            img = imresize(img, (128, 128))
            # imsave(f"{filename}_shrink.png", img)
            answer = model.predict([img.astype(float)/255.])
            answers.append(answer)

        return answers

    def sort_into_folders(self, images):
        for answer in self.analyse(images):
            os.makedirs("good", exist_ok=True)
            os.makedirs("bad", exist_ok=True)

            if np.argmax(answer) == 1:
                shutil.copy(filename, "good")
            else:
                shutil.copy(filename, "bad")

            label = ["bad", "good"]
            print(label[np.argmax(answer)], answer, filename)

    def is_good(self, image):
        return np.argmax(self.analyse([image])[0]) == 1

    def is_bad(self, image):
        return not self.is_good()


# TODO: startup is slow, speed it up somehow
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("this program takes arguments: the filenames of the files to be classified")
        sys.exit(1)

    c = Classifier()
    c.sort_into_folders(sys.argv[1:])
