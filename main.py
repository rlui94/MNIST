"""
This program contains a perceptron learning algorithm trained on the MNIST database.
Code for extracting MNIST images based on https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""

from mnist.loader import MNIST
import struct
import random
import numpy as np

ETA = 0.1

if __name__ == '__main__':
    filename = {
        'train_images': './images/train-images-idx3-ubyte',
        'train_labels': './images/train-labels-idx1-ubyte',
        'test_images': './images/t10k-images-idx3-ubyte',
        'test_labels': './images/t10k-labels-idx1-ubyte'
    }

    mndata = MNIST('./images/')
    train_images, train_labels = mndata.load_training()
    #test_images, test_labels = mndata.load_testing()
    np_train = np.asarray(train_images, dtype=np.float32).reshape(60000, 28, 28) / 255
    bias = np.ones((60000, 28), dtype=np.float32)
    weights = np.full((29,), ETA)



