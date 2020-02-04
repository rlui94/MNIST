"""
This program contains a perceptron learning algorithm trained on the MNIST database.
Code for extracting MNIST images based on https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""

import data
from mnist.loader import MNIST
import random
import datetime
import numpy as np

ETA = 0.01
LOGFILE = 'eta01.txt'
CLASSES = 10


def train_on_set(train_data, test_data, weights, learning_rate, epochs):
    with open(LOGFILE, 'a') as file:
        file.write("%s: Beginning training.\n" % (datetime.datetime.now()))
        print("%s: Beginning training.\n" % (datetime.datetime.now()))
        prev_acc = check_accuracy(train_data.images, train_data.labels, weights, train_data.size)
        file.write("%s: Current accuracy is %.5f/%.5f.\n" % (datetime.datetime.now(), prev_acc, check_accuracy(test_data.images, test_data.labels, weights, 10000)))
        print("%s: Current accuracy is %.5f.\n" % (datetime.datetime.now(), prev_acc))
        for e in range(0, epochs):
            # file.write("%s: Beginning epoch %d.\n" % (datetime.datetime.now(), e+1))
            print("%s: Beginning epoch %d.\n" % (datetime.datetime.now(), e + 1))
            for n in range(0, train_data.size):
                if n % 50 == 0:
                    acc = check_accuracy(train_data.images, train_data.labels, weights, train_data.size)
                    test_acc = check_accuracy(test_data.images, test_data.labels, weights, test_data.size)
                    # file.write("%s: The accuracy for %dth input is %d.\n" % (datetime.datetime.now(), n, check_accuracy(train_data.images, train_data.labels, weights, size)))
                    print("%s: The accuracy for %dth input is %.5f/%.5f.\n" % (
                    datetime.datetime.now(), n, acc, test_acc))
                    if acc > 0.85:
                        make_conf_matrix(test_data.images, test_data.labels, weights, test_data.size)
                perceptrons = np.zeros(10)
                for i in range(0, CLASSES):
                    perceptrons[i] = np.dot(train_data.images[n], weights[i])
                prediction = np.argmax(perceptrons)
                threshold = np.where(perceptrons > 0, 1, 0)
                if prediction != train_data.labels[n]:
                    for i in range(0, CLASSES):
                        for j in range(0, len(weights[i])):
                            if i == train_data.labels[n]:
                                weights[i, j] -= learning_rate * (threshold[i] - 1) * train_data.images[n, j]
                            else:
                                weights[i, j] -= learning_rate * (threshold[i] - 0) * train_data.images[n, j]
            # file.write("%s: Epoch %d complete.\n" % (datetime.datetime.now(), e+1))
            accuracy = check_accuracy(train_data.images, train_data.labels, weights, train_data.size)
            test_acc = check_accuracy(test_data.images, test_data.labels, weights, test_data.size)
            file.write("%s:%d, %.5f, %.5f.\n" % (datetime.datetime.now(), e+1, accuracy, test_acc))
            print("%s: Epoch %d complete.\n" % (datetime.datetime.now(), e + 1))
            print("%s: Accuracy for epoch %d is %.5f.\n" % (datetime.datetime.now(), e + 1, accuracy))
            if abs(prev_acc-accuracy)*100 < 0.01 or accuracy > 0.85:
                break


def check_accuracy(images, labels, weights, size):
    perceptrons = np.zeros(10)
    correct = 0
    for n in range(0, size):
        for i in range(0, CLASSES):
            perceptrons[i] = np.dot(images[n], weights[i])
        prediction = np.argmax(perceptrons)
        if prediction == labels[n]:
            correct += 1
    return correct/size

def make_conf_matrix(images, labels, weights, size):
    perceptrons = np.zeros(10)
    matrix = np.zeros((10, 10))
    for n in range(0, size):
        for i in range(0, CLASSES):
            perceptrons[i] = np.dot(images[n], weights[i])
        prediction = np.argmax(perceptrons)
        matrix[prediction, labels[n]] += 1
    return print(matrix)


if __name__ == '__main__':
    mndata = MNIST('./images/')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    train_data = data.Data()
    train_data.load(60000, train_images, train_labels)
    test_data = data.Data()
    test_data.load(10000, test_images, test_labels)
    weights = np.random.rand(10, 785) - .5
    train_on_set(train_data, test_data, weights, 60000, ETA, 70)
    make_conf_matrix(test_data.images, test_labels, weights, 10000)





