'''
Created on Dec 30, 2015

@author: Aaron Klein
'''
import time
import numpy as np
from sklearn import svm

from robo.task.base_task import BaseTask


class SVM(BaseTask):

    def __init__(self, train, train_targets, valid, valid_targets):
        X_lower = np.array([-10, -10])
        X_upper = np.array([10, 10])

        self.train = train
        self.train_targets = train_targets
        self.valid = valid
        self.valid_targets = valid_targets

        super(SVM, self).__init__(X_lower, X_upper)

    def objective_function(self, x):

        C = np.exp(float(x[0, 0]))
        gamma = np.exp(float(x[0, 1]))

        clf = svm.SVC(gamma=gamma, C=C)

        clf.fit(self.train, self.train_targets)
        y = 1 - clf.score(self.valid, self.valid_targets)
        y = np.log(y)
        return np.array([[y]])

    def objective_function_test(self, x):
        return self.objective_function(x)


class EnvSVM(BaseTask):

    def __init__(self, train, train_targets,
                 valid, valid_targets, with_costs=True):

        self.svm = SVM(train, train_targets, valid, valid_targets)
        #Use 10 time the number of classes as lower bound
        self.n_classes = np.unique(self.svm.train_targets).shape[0]

        self.s_min = np.log(10 * self.n_classes)
        self.s_max = np.log(self.svm.train_targets.shape[0])

        X_lower = np.concatenate((self.svm.original_X_lower,
                                  np.array([self.s_min])))
        X_upper = np.concatenate((self.svm.original_X_upper,
                                  np.array([self.s_max])))

        self.is_env = np.zeros([self.svm.n_dims])
        self.is_env = np.concatenate((self.is_env, np.array([1])))

        self.with_costs = with_costs

        super(EnvSVM, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        s = time.time()
        size = int(np.exp(x[0, -1]))

        print "Dataset size: %d" % size

        shuffle = np.random.permutation(np.arange(int(np.exp(self.s_max))))

        train = self.svm.train[shuffle[:size]]
        train_targets = self.svm.train_targets[shuffle[:size]]

        i = 0
        # Check if we have a sample of each class in the subset
        while True:
            if (np.unique(train_targets).shape[0] == self.n_classes):
                break
            shuffle = np.random.permutation(np.arange(int(np.exp(self.s_max))))
            train = self.svm.train[shuffle[:size]]
            train_targets = self.svm.train_targets[shuffle[:size]]
            i += 1
            # Sanity check if we can actually find a valid shuffled split
            if i == 20:
                raise("Couldn't find a valid split that contains a \
                sample from each class after 20 iterations. \
                Maybe increase your bounds!")

        C = np.exp(float(x[0, 0]))
        gamma = np.exp(float(x[0, 1]))

        clf = svm.SVC(gamma=gamma, C=C)

        clf.fit(train, train_targets)
        y = 1 - clf.score(self.svm.valid, self.svm.valid_targets)
        c = time.time() - s

        if self.with_costs:
            y = np.log(y)
            return np.array([[y]]), np.array([[c]])
        else:
            y = np.log(y)
            return np.array([[y]])

    def objective_function_test(self, x):
        return self.svm.objective_function_test(x[:, :-1])


class MultiSVM(BaseTask):
    '''
    classdocs
    '''

    def __init__(self, train, train_targets, valid, valid_targets):
        self.svm = SVM(train, train_targets, valid, valid_targets)

        # Add dimension for the tasks
        X_lower = np.concatenate((self.svm.original_X_lower,
                                  np.array([0])))
        X_upper = np.concatenate((self.svm.original_X_upper,
                                  np.array([1])))

        self.is_env = np.zeros([self.svm.n_dims])
        self.is_env = np.concatenate((self.is_env, np.array([1])))

        # Take 20 % of the data as subset for the smaller task
        self.subset_size = int(self.svm.train.shape[0] * 0.2)
        super(MultiSVM, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        s = time.time()

        # Evaluate only of a subset
        if np.round(x[0, -1]) == 0:

            train = self.svm.train[:self.subset_size]
            train_targets = self.svm.train_targets[:self.subset_size]

            C = np.exp(float(x[0, 0]))
            gamma = np.exp(float(x[0, 1]))

            clf = svm.SVC(gamma=gamma, C=C)

            clf.fit(train, train_targets)
            y = 1 - clf.score(self.svm.valid, self.svm.valid_targets)
            y = np.log(y)
            y = np.array([[y]])
        # Evaluate on whole data set
        elif np.round(x[0, -1]) == 1:

            y = self.svm.objective_function(x[:, :-1])

        c = time.time() - s

        return y, np.array([[c]])

    def objective_function_test(self, x):
        return self.svm.objective_function_test(x[:, :-1])
