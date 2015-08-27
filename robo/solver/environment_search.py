'''
Created on Jun 11, 2015

@author: Aaron Klein
'''

import time
import logging
import numpy as np

from robo.solver.bayesian_optimization import BayesianOptimization
from robo.recommendation.optimize_posterior import env_optimize_posterior_mean_and_std,\
    env_optimize_posterior_mean_and_std_mcmc


class EnvironmentSearch(BayesianOptimization):

    def __init__(self, acquisition_fkt=None, model=None, cost_model=None, maximize_fkt=None,
                 task=None, save_dir=None, initialization=None, recommendation_strategy=env_optimize_posterior_mean_and_std, num_save=1):

        logging.basicConfig(level=logging.DEBUG)
        # Initialize everything that is necessary
        self.acquisition_fkt = acquisition_fkt
        self.model = model
        self.maximize_fkt = maximize_fkt
        self.task = task

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
        # How often we restart the local search to find the current incumbent
        self.n_restarts = 19
        super(EnvironmentSearch, self).__init__(acquisition_fkt, model, maximize_fkt, task, save_dir)

    def initialize(self):
        start_time = time.time()
        super(EnvironmentSearch, self).initialize()
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
            logging.info("Start iteration %d ... ", it)
            # Choose a new configuration
            start_time = time.time()
            new_x = self.choose_next(self.X, self.Y, self.Costs)
            time_optimization_overhead = time.time() - start_time
            self.time_optimization_overhead = np.append(self.time_optimization_overhead, np.array([time_optimization_overhead]))
            logging.info("Optimization overhead was %f seconds" % (self.time_optimization_overhead[-1]))

            # Evaluate the configuration
            logging.info("Evaluate candidate %s" % (str(new_x)))
            start_time = time.time()
            new_y = self.task.evaluate(np.array(new_x))
            time_func_eval = time.time() - start_time
            new_cost = np.array([time_func_eval])
            self.time_func_eval = np.append(self.time_func_eval, np.array([time_func_eval]))
            logging.info("Configuration achieved a performance of %f in %s seconds" % (new_y[0, 0], new_cost[0]))

            # Update the data
            self.X = np.append(self.X, new_x, axis=0)
            self.Y = np.append(self.Y, new_y, axis=0)
            self.Costs = np.append(self.Costs, new_cost[:, np.newaxis], axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                self.save_iteration(it, costs=self.Costs[-1])

        # Recompute the incumbent before we return it
        self.estimate_incumbent()

        logging.info("Return %s as incumbent" % (str(self.incumbent)))
        return self.incumbent

    def estimate_incumbent(self):
        # Estimate new incumbent from the updated posterior
        if self.recommendation_strategy == None:
            best_idx = np.argmin(self.Y)
            self.incumbent = self.X[best_idx]
            self.incumbent_value = self.Y[best_idx]
        else:
            # Start one local search from the best observed point and N - 1 from random points
            startpoints = [np.random.uniform(self.task.X_lower, self.task.X_upper, self.task.n_dims) for i in range(self.n_restarts)]
            best_idx = np.argmin(self.Y)
            startpoints.append(self.X[best_idx])
            # Project startpoints to the configuration space
            for startpoint in startpoints:
                startpoint[self.task.is_env == 1] = self.task.X_upper[self.task.is_env == 1]

            if self.recommendation_strategy == env_optimize_posterior_mean_and_std or self.recommendation_strategy == env_optimize_posterior_mean_and_std_mcmc:
                self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model,
                                                                                    self.task.X_lower,
                                                                                    self.task.X_upper,
                                                                                    self.task.is_env,
                                                                                    startpoints,
                                                                                    with_gradients=True)
            else:
                logging.error("Recommendation strategy %s does not work for environment search." % self.recommendation_strategy)
                #self.incumbent, self.incumbent_value = self.recommendation_strategy(self.model, self.task.X_lower, self.task.X_upper, self.task.is_env, startpoint)

    def choose_next(self, X=None, Y=None, Costs=None):
        if X is not None and Y is not None:
            # Train the model for the objective function as well as for the cost function
            try:
                t = time.time()
                self.model.train(X, Y)
                self.cost_model.train(X, Costs)
                logging.info("Time to train the models: %f", (time.time() - t))
            except Exception, e:
                logging.error("Model could not be trained with data:", X, Y, Costs)
                raise
            self.model_untrained = False

            # Update the acquisition function with the new models
            self.acquisition_fkt.update(self.model, self.cost_model)

            # Estimate new incumbent from the updated posterior
            self.estimate_incumbent()

            # Maximize the acquisition function and return the suggested point
            t = time.time()
            x = self.maximize_fkt.maximize()
            logging.info("Time to maximize the acquisition function: %f", (time.time() - t))
        else:
            self.initialize()
            x = self.X
        return x
