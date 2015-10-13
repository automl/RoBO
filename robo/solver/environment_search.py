'''
Created on Jun 11, 2015

@author: Aaron Klein
'''
import sys
import time
import logging
import numpy as np
import StringIO
import cma

from robo.models.gpy_model import GPyModel
from robo.solver.bayesian_optimization import BayesianOptimization
#from robo.recommendation.optimize_posterior import env_optimize_posterior_mean_and_std


class EnvironmentSearch(BayesianOptimization):

    def __init__(self, acquisition_func=None, model=None, cost_model=None, maximize_func=None,
                 task=None, save_dir=None, initialization=None,
                 num_save=1, synthetic_func=False, train_intervall=1):

        logging.basicConfig(level=logging.DEBUG)

        self.train_intervall = train_intervall
        self.acquisition_func = acquisition_func
        self.model = model
        self.maximize_func = maximize_func
        self.task = task
        self.synthetic_func = synthetic_func

        self.initialization = initialization
        self.cost_model = cost_model
        self.save_dir = save_dir
        self.num_save = num_save
        if save_dir is not None:
            self.create_save_dir()
            
        self.n_init_points=3 


        self.X = None
        self.Y = None
        self.Costs = None
        self.model_untrained = True
        self.incumbent = None
        # How often we restart the local search to find the current incumbent
        self.n_restarts = 10
        super(EnvironmentSearch, self).__init__(acquisition_func, model, maximize_func, task, save_dir)

    def env_optimize_posterior_mean_and_std(self, model, X_lower, X_upper, is_env, startpoint):
        
        # We only optimize the posterior in the projected subspace
        env_values = X_upper[is_env == 1]
        sub_X_lower = X_lower[is_env == 0]
        sub_X_upper = X_upper[is_env == 0]
    
        def f(x):
            # Project x to the subspace
            x_ = np.zeros([is_env.shape[0]])
            x_[is_env == 1] = env_values
            x_[is_env == 0] = x
    
            mu, var = model.predict(x_[np.newaxis, :])
            return (mu + np.sqrt(var))[0, 0]

        stdout = sys.stdout
        sys.stdout = StringIO.StringIO()
    
        res = cma.fmin(f, startpoint[is_env == 0], 0.6, options={"bounds": [sub_X_lower, sub_X_upper]})
        xopt = np.zeros([is_env.shape[0]])
        xopt[is_env == 1] = env_values
        xopt[is_env == 0] = res[0]
        fval = res[1]
        sys.stdout = stdout    
        return xopt, fval 

    def initialize(self):
        #super(EnvironmentSearch, self).initialize()
        #
        
        self.time_func_eval = np.zeros([self.n_init_points])
        self.time_optimization_overhead = np.zeros([self.n_init_points])
        self.X = np.zeros([1, self.task.n_dims])
        self.Y = np.zeros([1, 1])
        self.Costs = np.zeros([1, 1])

        #grid = []
        #grid.append(np.array(self.task.X_lower))
        #grid.append(np.array(self.task.X_upper))
        #grid.append(np.array((self.task.X_upper - self.task.X_lower) / 2))
        #grid = np.array(grid)

        for i in range(self.n_init_points):
        #for i, x in enumerate(grid):
            start_time = time.time()
            #TODO: Sample random points in subspace
            x = np.array([np.random.uniform(self.task.X_lower, self.task.X_upper, self.task.n_dims)])
            self.time_optimization_overhead[i] = time.time() - start_time
            logging.info("Evaluate: %s" % x)
            #x = x[np.newaxis, :]

            start_time = time.time()
            y = self.task.evaluate(x)
            self.time_func_eval[i] = time.time() - start_time


            if self.synthetic_func:
                #self.Costs[i] = np.exp(x[:, self.task.is_env == 1])[0]
                new_cost = 1. / (np.e - 1) * (np.exp(x[:, self.task.is_env == 1])[0] - 1)
            else:
                new_cost = np.array([time.time() - start_time])
                
            
            if i == 0:
                self.X[i] = x[0, :]
                self.Y[i] = y[0, :]
                self.Costs[i] = new_cost
            else:
                self.X = np.append(self.X, x, axis=0)
                self.Y = np.append(self.Y, y, axis=0)
                self.Costs = np.append(self.Costs, new_cost[np.newaxis, :], axis=0)

            # Use best point seen so far as incumbent
            best_idx = np.argmin(self.Y)
            self.incumbent = self.X[best_idx]
            self.incumbent_value = self.Y[best_idx]
            self.incumbent[self.task.is_env == 1] = self.task.X_upper[self.task.is_env == 1]

            if self.save_dir is not None and (i) % self.num_save == 0:
                self.save_iteration(i, costs=self.Costs, hyperparameters=None, acquisition_value=0)

    def run(self, num_iterations=10, X=None, Y=None, Costs=None):

        self.time_start = time.time()

        if X is None and Y is None:
            # No data yet start with initialization procedure
            self.initialize()

        else:
            self.X = X
            self.Y = Y
            self.Costs = Costs

        for it in range(self.n_init_points, num_iterations):
            logging.info("Start iteration %d ... ", it)
            # Choose a new configuration
            start_time = time.time()
            if it % self.train_intervall == 0:
                do_optimize = True
            else:
                do_optimize = False
            new_x = self.choose_next(self.X, self.Y, self.Costs, do_optimize)

            # Estimate current incumbent from our posterior
            start_time_inc = time.time()
            self.estimate_incumbent()
            logging.info("New incumbent %s found in %f seconds", str(self.incumbent), time.time() - start_time_inc)

            time_optimization_overhead = time.time() - start_time
            self.time_optimization_overhead = np.append(self.time_optimization_overhead, np.array([time_optimization_overhead]))
            logging.info("Optimization overhead was %f seconds" % (self.time_optimization_overhead[-1]))

            # Evaluate the configuration
            logging.info("Evaluate candidate %s" % (str(new_x)))
            start_time = time.time()
            new_y = self.task.evaluate(new_x)
            time_func_eval = time.time() - start_time
            new_cost = np.array([time_func_eval])
            ############################################ Debugging ############################################
            if self.synthetic_func:
                logging.info("Optimizing a synthetic functions for that we use np.exp(x[-1]) as cost!")
                #new_cost = np.exp(new_x[:, self.task.is_env == 1])[0]
                new_cost = 1. / (np.e - 1) * (np.exp(new_x[:, self.task.is_env == 1])[0] - 1)
            self.time_func_eval = np.append(self.time_func_eval, np.array([time_func_eval]))
            logging.info("Configuration achieved a performance of %f in %s seconds" % (new_y[0, 0], new_cost[0]))

            # Update the data
            self.X = np.append(self.X, new_x, axis=0)
            self.Y = np.append(self.Y, new_y, axis=0)
            self.Costs = np.append(self.Costs, new_cost[:, np.newaxis], axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                if isinstance(self.model, GPyModel):
                    hypers = self.model.m.param_array
                else:
                    hypers = None
                    
                self.save_iteration(it, costs=self.Costs[-1], hyperparameters=hypers, acquisition_value=self.acquisition_func(new_x))

        logging.info("Return %s as incumbent" % (str(self.incumbent)))
        return self.incumbent

    def estimate_incumbent(self):
        # Start one local search from the best observed point and N - 1 from random points
        #startpoints = [np.random.uniform(self.task.X_lower, self.task.X_upper, self.task.n_dims) for i in range(self.n_restarts)]
        #best_idx = np.argmin(self.Y)
        #startpoints.append(self.X[best_idx])
        # Project startpoints to the configuration space
        #for startpoint in startpoints:
        #    startpoint[self.task.is_env == 1] = self.task.X_upper[self.task.is_env == 1]

        #self.incumbent, self.incumbent_value = env_optimize_posterior_mean_and_std(self.model,
        #                                                                    self.task.X_lower,
        #                                                                    self.task.X_upper,
        #                                                                    self.task.is_env,
        #                                                                    startpoints,
        #                                                                    with_gradients=True)
        
        best_idx = np.argmin(self.Y)
        startpoints = self.X[best_idx]
        # Project startpoints to the configuration space
        startpoints[self.task.is_env == 1] = self.task.X_upper[self.task.is_env == 1]
        self.incumbent, self.incumbent_value = self.env_optimize_posterior_mean_and_std(self.model,
                                                                            self.task.X_lower,
                                                                            self.task.X_upper,
                                                                            self.task.is_env,
                                                                            startpoints)

    def choose_next(self, X=None, Y=None, Costs=None, do_optimize=True):
        if X is not None and Y is not None:
            # Train the model for the objective function as well as for the cost function
            try:
                t = time.time()
                self.model.train(X, Y, do_optimize)
                #self.cost_model.train(X[:, self.task.is_env == 1], Costs, do_optimize)
                self.cost_model.train(X, Costs, do_optimize)

                logging.info("Time to train the models: %f", (time.time() - t))
            except Exception, e:
                logging.error("Model could not be trained with data:", X, Y, Costs)
                raise
            self.model_untrained = False

            # Update the acquisition function with the new models
            self.acquisition_func.update(self.model, self.cost_model)

            # Maximize the acquisition function and return the suggested point
            t = time.time()
            x = self.maximize_func.maximize()
            logging.info("Time to maximize the acquisition function: %f", (time.time() - t))
        else:
            self.initialize()
            x = self.X
        return x
