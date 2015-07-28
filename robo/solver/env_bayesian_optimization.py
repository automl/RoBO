'''
Created on Jun 11, 2015

@author: Aaron Klein
'''

import os
import time
import shutil
import errno
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

from robo.solver.bayesian_optimization import BayesianOptimization
from robo.recommendation.optimize_posterior import env_optimize_posterior_mean_and_std


class EnvBayesianOptimization(BayesianOptimization):

    def __init__(self, acquisition_fkt=None, model=None, cost_model=None, maximize_fkt=None,
                 task=None, save_dir=None, initialization=None, recommendation_strategy=None, num_save=1):

        # Initialize all members
        self.initialization = initialization
        self.task = task
        self.acquisition_fkt = acquisition_fkt
        self.model = model
        self.cost_model = cost_model
        self.maximize_fkt = maximize_fkt
        self.save_dir = save_dir
        self.num_save = num_save
        if save_dir is not None:
            self.create_save_dir()

        self.X = None
        self.Y = None
        self.Costs = None
        self.model_untrained = True
        self.recommendation_strategy = recommendation_strategy
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
            new_y = self.task.evaluate(np.array(new_x))
            new_cost = np.array([time.time() - start])

            print "Configuration achieved a performance of %f in %s seconds" % (new_y[0, 0], new_cost[0])

            self.X = np.append(self.X, new_x, axis=0)
            self.Y = np.append(self.Y, new_y, axis=0)
            self.Costs = np.append(self.Costs, new_cost[:, np.newaxis], axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                self.save_iteration(it)

        # Recompute the incumbent before we return it
        if self.recommendation_strategy == None:
            best_idx = np.argmin(self.Y)
            self.incumbent = self.X[best_idx]
            self.incumbent_value = self.Y[best_idx]
        else:
            if self.recommendation_strategy is env_optimize_posterior_mean_and_std:
                    startpoint = np.array([np.random.uniform(self.task.X_lower[self.task.is_env == 0],
                                                             self.task.X_upper[self.task.is_env == 0],
                                                             self.task.X_lower[self.task.is_env == 0].shape[0])])
                    print "Use startpoint %s" % (startpoint)
                    self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model,
                                                                                        self.task.X_lower[self.task.is_env == 0],
                                                                                        self.task.X_upper[self.task.is_env == 0],
                                                                                        self.task.is_env,
                                                                                        self.task.X_upper[self.task.is_env == 1],
                                                                                        inc=startpoint)
            else:
                self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper)

        print "Return %s as incumbent" % (str(self.incumbent))
        return self.incumbent

    def choose_next(self, X=None, Y=None, Costs=None):
        if X is not None and Y is not None:
            try:
                self.model.train(X, Y)
                self.cost_model.train(X, Costs)
                print "Trained"
            except Exception, e:
                print "Model could not be trained with data:", X, Y, Costs
                raise
            self.model_untrained = False
            self.acquisition_fkt.update(self.model, self.cost_model)

            #TODO: change default strategy
            if self.recommendation_strategy == None:
                best_idx = np.argmin(self.Y)
                self.incumbent = self.X[best_idx]
                self.incumbent_value = self.Y[best_idx]
            else:
                if self.recommendation_strategy is env_optimize_posterior_mean_and_std:
                    startpoint = np.array([np.random.uniform(self.task.X_lower[self.task.is_env == 0],
                                                             self.task.X_upper[self.task.is_env == 0],
                                                             self.task.X_lower[self.task.is_env == 0].shape[0])])
                    print "Use startpoint %s" % (startpoint)
                    self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model,
                                                                                        self.task.X_lower[self.task.is_env == 0],
                                                                                        self.task.X_upper[self.task.is_env == 0],
                                                                                        self.task.is_env,
                                                                                        self.task.X_upper[self.task.is_env == 1],
                                                                                        inc=startpoint)
                else:
                    self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper)

            x = self.maximize_fkt.maximize()
        else:
            self.initialize()
            x = self.X
        return x

    def save_iteration(self, it):
        """
            Saves an iteration.
        """
        file_name = "iteration_%03d.pkl" % (it)

        file_name = os.path.join(self.save_dir, file_name)
        #os.makedirs(iteration_folder)
        #FIXME What does Entropy return as incumbent?
        #if hasattr(self.acquisition_fkt, "_get_most_probable_minimum") and not self.model_untrained:
        #    pickle.dump([self.X, self.Y, self.acquisition_fkt._get_most_probable_minimum()[0], self.time_func_eval, self.time_optimization_overhead], open(file_name, "w"))
        #else:
        pickle.dump([self.X, self.Y, self.Costs, self.incumbent, self.incumbent_value, self.time_func_eval, self.time_optimization_overhead], open(file_name, "w"))
