'''
Created on Jul 28, 2015

@author: Aaron Klein
'''
import time
import cPickle
import numpy as np

from robo.task.base_task import BaseTask

from sklearn.ensemble import RandomForestRegressor


class SurrogateLeNetMnist(BaseTask):

    def __init__(self):

        X_lower = np.array([0.0, 0.0, 0.00001, 0.5])
        X_upper = np.array([0.9, 0.9, 0.1, 0.9])
        super(SurrogateLeNetMnist, self).__init__(X_lower, X_upper)

    def objective_function(self, x):

        rf = cPickle.load(open("/mhome/kleinaa/experiments/entropy_search/surrogates/lenet/rf.pkl", "r"))

        validation_error = rf.predict(x)
        return validation_error[:, np.newaxis]

    def evaluate_test(self, x):
        return self.objective_function(x)


class SurrogateEnvLeNetMnist(BaseTask):

    def __init__(self):

        X_lower = np.array([0.0, 0.0, 0.00001, 0.5, 6.91])
        X_upper = np.array([0.9, 0.9, 0.1, 0.9, 10.81978])
        self.is_env = np.array([0, 0, 0, 0, 1])
        super(SurrogateEnvLeNetMnist, self).__init__(X_lower, X_upper)

    def objective_function(self, x):

        rf = cPickle.load(open("/mhome/kleinaa/experiments/entropy_search/surrogates/env_lenet/rf.pkl", "r"))
        validation_error = rf.predict(x)
        time.sleep(x[0, -1])
        return validation_error[:, np.newaxis]

    def evaluate_test(self, x):
        return self.objective_function(x)
