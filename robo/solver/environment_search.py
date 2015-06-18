'''
Created on Jun 11, 2015

@author: Aaron Klein
'''

import random
import os
import errno
import numpy as np
import shutil
from robo.util.exc import BayesianOptimizationError
from robo.bayesian_optimization import BayesianOptimization

here = os.path.abspath(os.path.dirname(__file__))


class EnvironmentSearch(BayesianOptimization):

    def __init__(self, acquisition_fkt=None, model=None,
                 maximize_fkt=None, X_lower=None, X_upper=None, dims=None,
                 objective_fkt=None, save_dir=None, pcs_file=None, num_save=1):

        # Initialize all members

        # Parse pcs file to estimate which parameters are environment variables
        pass

    def initialize(self):
        pass

    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        pass

    def choose_next(self, X=None, Y=None):
        if X is not None and Y is not None:
            try:
                self.model.train(X, Y)
            except Exception, e:
                print "Model could not be trained", X, Y
                raise
            self.model_untrained = False
            self.acquisition_fkt.update(self.model)

            if self.recommendation_strategy is None:
                best_idx = np.argmin(Y)
                self.incumbent = X[best_idx]
            else:
                self.incumbent = self.recommendation_strategy(self.model, self.acquisition_fkt)

            x = self.maximize_fkt(self.acquisition_fkt, self.X_lower, self.X_upper)
        else:
            X = np.empty((1, self.dims))
            for i in range(self.dims):
                X[0, i] = random.random() * (self.X_upper[i] - self.X_lower[i]) + self.X_lower[i]
            x = np.array(X)
        return x
