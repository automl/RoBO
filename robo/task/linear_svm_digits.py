'''
Created on 12.07.2015

@author: Aaron Klein
'''
import numpy as np

from robo.task.base_task import BaseTask

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


class LinearSVMDigits(BaseTask):
    '''
    classdocs
    '''

    def __init__(self):
        self.X_lower = np.array([-5, 10])
        self.X_upper = np.array([15, 1347])
        self.n_dims = 2
        self.is_env = np.array([0, 1])

    def objective_function(self, x):
        digits = load_digits(n_class=10)
        data = digits["data"]
        target = digits["target"]

        data_train, data_test, target_train, target_test = train_test_split(data, target)
        data_train = data_train[:x[0, 1]]
        target_train = target_train[:x[0, 1]]
        svm = LinearSVC(C=(2 ** x[0, 0]))
        svm.fit(data_train, target_train)
        pred = svm.predict(data_test)
        return np.array([[1 - accuracy_score(target_test, pred)]])
