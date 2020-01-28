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
CLASSES = 10


def train_on_set(train_images, train_labels, weights, size, learning_rate, epochs):
    with open(LOGFILE, 'a') as file:
        file.write("%s: Beginning training.\n" % (datetime.datetime.now()))
        file.write("%s: Current accuracy is %d.\n" % (datetime.datetime.now(), check_accuracy(train_images, train_labels, weights, size)))
        for e in range(0, epochs):
            file.write("%s: Beginning epoch %d.\n" % (datetime.datetime.now(), e+1))
            perceptrons = np.zeros(10)
            for n in range(0, size):
                for i in range(0, CLASSES):
                    perceptrons[i] = np.dot(train_images[n], weights[i])
                thresholds = np.where(perceptrons > 0, 1, 0)
                prediction = np.argmax(thresholds)
                if prediction != train_labels[n]:
                    for i in range(0, CLASSES):
                        for j in enumerate(weights[i]):
                            weights[i][j] -= learning_rate * (prediction - train_labels[n]) * train_images[n][j]
            file.write("%s: Epoch %d complete.\n" % (datetime.datetime.now(), e+1))
            file.write("%s: Accuracy for epoch %d is %d.\n" % (
            datetime.datetime.now(), e+1, check_accuracy(train_images, train_labels, weights, size)))


def check_accuracy(train_images, train_labels, weights, size):
    perceptrons = np.zeros(10)
    correct = 0
    for n in range(0, size):
        for i in range(0, CLASSES):
            perceptrons[i] = np.dot(train_images[n], weights[i])
        thresholds = np.where(perceptrons > 0, 1, 0)
        prediction = np.argmax(thresholds)
        if prediction == train_labels[n]:
            correct += 1
    return correct/size


if __name__ == '__main__':
    mndata = MNIST('./images/')
    train_images, train_labels = mndata.load_training()
    #test_images, test_labels = mndata.load_testing()
    np_train = np.asarray(train_images, dtype=np.float16).reshape(60000, 784) / 255  # divide to avoid huge weights
    bias = np.ones((60000, 1), dtype=np.float16)
    np_train = np.dstack((np_train, bias))  # concatenate matrix of 1s for bias
    weights = np.full((785,), ETA)  # CHANGE THIS LATER
    train_on_set(train_images, train_labels, weights, 5, ETA, 5)






