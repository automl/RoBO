'''
Created on Jun 11, 2015

@author: Aaron Klein
'''

import time
import shutil
import errno
import numpy as np

from robo.bayesian_optimization import BayesianOptimization


class EnvBayesianOptimization(BayesianOptimization):

    def __init__(self, acquisition_fkt=None, model=None, cost_model=None,
                 maximize_fkt=None, X_lower=None, X_upper=None, dims=None,
                 objective_fkt=None, save_dir=None, initialization=None, num_save=1):

        # Initialize all members
        self.initialization = initialization
        self.objective_fkt = objective_fkt
        self.acquisition_fkt = acquisition_fkt
        self.model = model
        self.cost_model = cost_model
        self.maximize_fkt = maximize_fkt
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.dims = dims
        self.save_dir = save_dir
        self.num_save = num_save
        if save_dir is not None:
            self.create_save_dir()

        self.X = None
        self.Y = None
        self.Costs = None
        self.model_untrained = True
        self.recommendation_strategy = None
        self.incumbent = None

    def initialize(self):
        start_time = time.time()
        super(EnvBayesianOptimization, self).initialize()
        self.Costs = np.array([[time.time() - start_time]])

    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        """
        overwrite:
            True: data present in save_dir will be deleted.
            False: data present will be loaded an the run will continue
        X, Y:
            Initial observations. They are optional. If a run continues
            these observations will be overwritten by the load
        """
        def _onerror(dirs, path, info):
            if info[1].errno != errno.ENOENT:
                raise

        if overwrite and self.save_dir:
            shutil.rmtree(self.save_dir, onerror=_onerror)
            self.create_save_dir()
        else:
            self.create_save_dir()

        if X is None and Y is None:
            self.initialize()
            num_iterations = num_iterations - 1
        else:
            self.X = X
            self.Y = Y
            # TODO: allow different initialization strategies here
            self.initialize()

        for it in range(num_iterations):
            print "Choose a new configuration"
            new_x = self.choose_next(self.X, self.Y, self.Costs)
            print "Evaluate candidate %s" % (str(new_x))

            start = time.time()
            new_y = self.objective_fkt(np.array(new_x))
            new_cost = np.array([time.time() - start])

            print "Configuration achieved a performance of %d in %s seconds" % (new_y[0, 0], new_cost[0])
            self.X = np.append(X, new_x, axis=0)
            self.Y = np.append(Y, new_y, axis=0)
            self.Costs = np.append(self.Costs, new_cost, axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                self.save_iteration(X, Y, new_x)

        # Recompute the incumbent before we return it
        if self.recommendation_strategy is None:
            best_idx = np.argmin(Y)
            self.incumbent = X[best_idx]
        else:
            self.incumbent = self.recommendation_strategy(self.model, self.acquisition_fkt)

        print "Return %s as incumbent" % (str(self.incumbent))
        return self.incumbent

    def choose_next(self, X=None, Y=None, Costs=None):
        if X is not None and Y is not None:
            try:
                self.model.train(X, Y)
                self.cost_model.train(X, Costs)
            except Exception, e:
                print "Model could not be trained with data:", X, Y
                raise
            self.model_untrained = False
            self.acquisition_fkt.update(self.model, self.cost_model)

            #TODO: change default strategy
            if self.recommendation_strategy is None:
                best_idx = np.argmin(Y)
                self.incumbent = X[best_idx]
            else:
                self.incumbent = self.recommendation_strategy(self.model, self.acquisition_fkt)

            x = self.maximize_fkt(self.acquisition_fkt, self.X_lower, self.X_upper)
        else:
            self.initialize()
            x = self.X
        return x
