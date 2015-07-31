'''
Created on Jul 28, 2015

@author: Aaron Klein
'''
import time
import cPickle
import numpy as np
from copy import deepcopy

from robo.task.base_task import BaseTask

from deep_nets.conv_nets.lenet import LeNet
from deep_nets.evaluation import train, test


class LeNetMnist(BaseTask):

    def __init__(self):

        X_lower = np.array([0.0, 0.0, 0.00001, 0.5])
        X_upper = np.array([0.9, 0.9, 0.1, 0.9])
        super(LeNetMnist, self).__init__(X_lower, X_upper)

        self.X_train = np.load("/home/kleinaa/data/mnist_npy/x_train.npy")
        self.X_test = np.load("/home/kleinaa/data/mnist_npy/x_test.npy")
        self.X_valid = np.load("/home/kleinaa/data/mnist_npy/x_valid.npy")
        self.y_train = np.load("/home/kleinaa/data/mnist_npy/y_train.npy")
        self.y_test = np.load("/home/kleinaa/data/mnist_npy/y_test.npy")
        self.y_valid = np.load("/home/kleinaa/data/mnist_npy/y_valid.npy")

    def objective_function(self, x):
        net = LeNet(dropout_fist_fc=float(x[0, 0]), dropout_second_fc=float(x[0, 1]), learning_rate=float(x[0, 2]), momentum=float(x[0, 3]))
        validation_error, _ = train(net, self.X_train, self.y_train, self.X_valid, self.y_valid, num_epochs=13)
        return np.array([[validation_error]])

    def evaluate_test(self, x):
        net = LeNet(dropout_fist_fc=float(x[0, 0]), dropout_second_fc=float(x[0, 1]), learning_rate=float(x[0, 2]), momentum=float(x[0, 3]))
        train(net, self.X_train, self.y_train, self.X_valid, self.y_valid, num_epochs=13)
        test_error = test(net, self.X_test, self.y_test)
        return np.array([[test_error]])


class EnvLeNetMnist(LeNetMnist):

    def __init__(self):

        # np.e ** 10.81978 = 50000
        super(EnvLeNetMnist, self).__init__()
        self.X_lower = np.concatenate((self.X_lower, np.array([6.91])))
        self.X_upper = np.concatenate((self.X_upper, np.array([10.81978])))
        self.n_dims = self.X_lower.shape[0]
        self.is_env = np.zeros([self.n_dims])
        self.is_env[-1] = 1

        #TODO: This might be a problem for large datasets
        self.X = deepcopy(self.X_train)
        self.y = deepcopy(self.y_train)

    def objective_function(self, x):
        self.X_train = self.X[:int(np.e ** x[0, -1])]
        self.y_train = self.y[:int(np.e ** x[0, -1])]

        return super(EnvLeNetMnist, self).objective_function(x)

    def evaluate_test(self, x):
        self.X_train = self.X
        self.y_train = self.y

        return super(EnvLeNetMnist, self).evaluate_test(x)
