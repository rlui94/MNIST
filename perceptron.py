"""
Perceptron object
"""

import numpy as np
import random


class Perceptron:
    def __init__(self, inputs):
        """
        Create a perceptron with randomized weights (including +1 for bias)
        :param inputs: number of inputs
        """
        self.weights = np.random.rand(inputs + 1) - .5

    def predict(self, inputs):
        """
        Return weighted sum of inputs. Does not squash.
        :param inputs: Input as numpy array
        :return: Weighted sum
        """
        return np.dot(self.weights, inputs)