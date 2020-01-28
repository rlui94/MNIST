"""
This program contains a perceptron learning algorithm trained on the MNIST database.
Code for extracting MNIST images based on https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""

from mnist.loader import MNIST
import struct
import random
import numpy as np

if __name__ == '__main__':
    filename = {
        'train_images': './images/train-images-idx3-ubyte',
        'train_labels': './images/train-labels-idx1-ubyte',
        'test_images': './images/t10k-images-idx3-ubyte',
        'test_labels': './images/t10k-labels-idx1-ubyte'
    }

    mndata = MNIST('./images/')
    images, labels = mndata.load_training()
    print(mndata.display(images[5]))
    print(labels[5])

