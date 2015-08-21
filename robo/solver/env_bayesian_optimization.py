'''
Created on Jun 11, 2015

@author: Aaron Klein
'''

import time
import logging
import numpy as np

from robo.solver.base_solver import BaseSolver
from robo.recommendation.optimize_posterior import env_optimize_posterior_mean_and_std,\
    env_optimize_posterior_mean_and_std_mcmc
from robo.recommendation.incumbent import compute_incumbent


class EnvBayesianOptimization(BaseSolver):

    def __init__(self, acquisition_fkt=None, model=None, cost_model=None, maximize_fkt=None,
                 task=None, save_dir=None, initialization=None, recommendation_strategy=None, num_save=1):

        logging.basicConfig(level=logging.DEBUG)
        # Initialize everything that is necessary
        super(EnvBayesianOptimization, self).__init__(acquisition_fkt, model, maximize_fkt, task, save_dir)

        self.initialization = initialization
        self.cost_model = cost_model
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

    def run(self, num_iterations=10, X=None, Y=None, Costs=None):

        self.time_start = time.time()

        if X is None and Y is None:
            # No data yet start with initialization procedure
            self.initialize()
            self.incumbent = self.X[0]
            self.incumbent_value = self.Y[0]
            if self.save_dir is not None and (0) % self.num_save == 0:
                self.save_iteration(0, costs=self.Costs)
        else:
            self.X = X
            self.Y = Y
            self.Costs = Costs

        for it in range(1, num_iterations):
            # Choose a new configuration
            new_x = self.choose_next(self.X, self.Y, self.Costs)

            # Evaluate the configuration
            start = time.time()
            new_y = self.task.evaluate(np.array(new_x))
            new_cost = np.array([time.time() - start])
            logging.info("Configuration achieved a performance of %f in %s seconds" % (new_y[0, 0], new_cost[0]))

            # Update the data
            self.X = np.append(self.X, new_x, axis=0)
            self.Y = np.append(self.Y, new_y, axis=0)
            self.Costs = np.append(self.Costs, new_cost[:, np.newaxis], axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                self.save_iteration(it, costs=self.Costs)

        # Recompute the incumbent before we return it
        if self.recommendation_strategy == None:
            best_idx = np.argmin(self.Y)
            self.incumbent = self.X[best_idx]
            self.incumbent_value = self.Y[best_idx]
        else:
            if self.recommendation_strategy is env_optimize_posterior_mean_and_std:
                    self._estimate_incumbent()
            else:
                self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper)

        logging.info("Return %s as incumbent" % (str(self.incumbent)))
        return self.incumbent

    def choose_next(self, X=None, Y=None, Costs=None):
        if X is not None and Y is not None:
            # Train the model for the objective function as well as for the cost function
            try:
                self.model.train(X, Y)
                self.cost_model.train(X, Costs)
            except Exception, e:
                logging.error("Model could not be trained with data:", X, Y, Costs)
                raise
            self.model_untrained = False

            # Update the acquisition function with the new models
            self.acquisition_fkt.update(self.model, self.cost_model)

            # Estimate new incumbent from the updated posterior
            if self.recommendation_strategy == None:
                best_idx = np.argmin(self.Y)
                self.incumbent = self.X[best_idx]
                self.incumbent_value = self.Y[best_idx]
            else:
                # Project best seen configuration to subspace and use it as startpoint
                startpoint, _ = compute_incumbent(self.model)
                startpoint[self.task.is_env == 1] = self.task.X_upper[self.task.is_env == 1]
                if isinstance(self.recommendation_strategy, env_optimize_posterior_mean_and_std) or isinstance(self.recommendation_strategy, env_optimize_posterior_mean_and_std_mcmc):
                    self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper, self.task.is_env, startpoint)
                else:
                    self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper, self.task.is_env, startpoint)

            # Maximize the acquisition function and return the suggested point
            x = self.maximize_fkt.maximize()
        else:
            self.initialize()
            x = self.X
        return x
