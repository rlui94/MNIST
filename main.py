"""
This program contains a perceptron learning algorithm trained on the MNIST database.
Code for extracting MNIST images based on https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""

from mnist.loader import MNIST
import random
import datetime
import numpy as np

ETA = 0.1
LOGFILE = 'log.txt'


def train_on_set(train_images, train_labels, weights, size, learning_rate, epochs):
    with open(LOGFILE, 'a') as file:
        for e in range(0, epochs):
            file.write("%s: Beginning epoch %d.\n Weights currently " % (datetime.datetime.now(), e+1))
            file.write(weights)
            file.write('\n')
            curr_img = 0
            correct = 0
            for i in range(0, size):
                activations = np.dot(train_images[i], weights)
                thresholds = np.where(activations > 0, 1, 0)

                curr_img += 1
            file.write("%s: Epoch %d complete. Current accuracy is %d.\n" % (datetime.datetime.now(), e+1, correct/curr_img))



if __name__ == '__main__':
    mndata = MNIST('./images/')
    train_images, train_labels = mndata.load_training()
    #test_images, test_labels = mndata.load_testing()
    np_train = np.asarray(train_images, dtype=np.float32).reshape(60000, 784) / 255  # divide to avoid huge weights
    bias = np.ones((60000, 784), dtype=np.float32)
    np_train = np.dstack((np_train, bias))  # concatenate matrix of 1s for bias
    weights = np.full((29,), ETA)  # CHANGE THIS LATER
    train_on_set(train_images, train_labels, weights, 5, ETA, 5)






