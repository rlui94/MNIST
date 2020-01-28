"""
This program contains a perceptron learning algorithm trained on the MNIST database.
Code for extracting MNIST images based on https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""

from mnist.loader import MNIST
import random
import datetime
import numpy as np

ETA = 0.001
LOGFILE = 'eta001.txt'
CLASSES = 10


def train_on_set(train_images, train_labels, test_images, test_labels, weights, size, learning_rate, epochs):
    with open(LOGFILE, 'a') as file:
        file.write("%s: Beginning training.\n" % (datetime.datetime.now()))
        print("%s: Beginning training.\n" % (datetime.datetime.now()))
        prev_acc = check_accuracy(train_images, train_labels, weights, size)
        file.write("%s: Current accuracy is %.5f.\n" % (datetime.datetime.now(), prev_acc))
        print("%s: Current accuracy is %.5f.\n" % (datetime.datetime.now(), prev_acc))
        for e in range(0, epochs):
            #file.write("%s: Beginning epoch %d.\n" % (datetime.datetime.now(), e+1))
            print("%s: Beginning epoch %d.\n" % (datetime.datetime.now(), e + 1))
            for n in range(0, size):
                if n%50==0:
                    #file.write("%s: The accuracy for %dth input is %d.\n" % (datetime.datetime.now(), n, check_accuracy(train_images, train_labels, weights, size)))
                    print("%s: The accuracy for %dth input is %.5f.\n" % (
                    datetime.datetime.now(), n, check_accuracy(train_images, train_labels, weights, size)))
                perceptrons = np.zeros(10)
                for i in range(0, CLASSES):
                    perceptrons[i] = np.dot(train_images[n], weights[i])
                prediction = np.argmax(perceptrons)
                threshold = np.where(perceptrons > 0, 1, 0)
                if prediction != train_labels[n]:
                    for i in range(0, CLASSES):
                        for j in range(0, len(weights[i])):
                            if i == train_labels[n]:
                                weights[i, j] -= learning_rate * (threshold[i] - 1) * train_images[n, j]
                            else:
                                weights[i, j] -= learning_rate * (threshold[i] - 0) * train_images[n, j]
            #file.write("%s: Epoch %d complete.\n" % (datetime.datetime.now(), e+1))
            accuracy = check_accuracy(train_images, train_labels, weights, size)
            test_acc = check_accuracy(test_images, test_labels, weights, 10000)
            file.write("%s:%d, %.5f, %.5f.\n" % (datetime.datetime.now(), e+1, accuracy, test_acc))
            print("%s: Epoch %d complete.\n" % (datetime.datetime.now(), e + 1))
            print("%s: Accuracy for epoch %d is %.5f.\n" % (datetime.datetime.now(), e + 1, accuracy))
            if abs(prev_acc-accuracy)*100 < 0.01:
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


if __name__ == '__main__':
    mndata = MNIST('./images/')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    np_train = np.asarray(train_images, dtype=np.float16).reshape(60000, 784) / 255  # divide to avoid huge weights
    np_test = np.asarray(test_images, dtype=np.float16).reshape(10000, 784) / 255
    bias = np.ones((60000, 1), dtype=np.float16)
    np_train = np.concatenate((np_train, bias), axis=1)  # concatenate matrix of 1s for bias
    bias = np.ones((10000, 1), dtype=np.float16)
    np_test = np.concatenate((np_test, bias), axis=1)
    weights = np.random.rand(10, 785) - .5
    train_on_set(np_train, train_labels, np_test, test_labels, weights, 60000, ETA, 70)






