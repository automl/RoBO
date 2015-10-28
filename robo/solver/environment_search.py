'''
Created on Jun 11, 2015

@author: Aaron Klein
'''

import time
import logging
import numpy as np

import cma

from robo.solver.bayesian_optimization import BayesianOptimization


logger = logging.getLogger(__name__)


class EnvironmentSearch(BayesianOptimization):

    def __init__(self, acquisition_func=None,
                 model=None,
                 cost_model=None,
                 maximize_func=None,
                 task=None,
                 save_dir=None,
                 initialization=None,
                 num_save=1,
                 train_intervall=1,
                 n_restarts=1,
                 n_init_points=15):

        self.train_intervall = train_intervall
        self.acquisition_func = acquisition_func
        self.model = model
        self.maximize_func = maximize_func
        self.task = task

        self.initialization = initialization
        self.cost_model = cost_model
        self.save_dir = save_dir
        self.num_save = num_save
        self.n_init_points = n_init_points

        if save_dir is not None:
            self.create_save_dir()

        self.X = None
        self.Y = None
        self.Costs = None
        self.model_untrained = True
        self.incumbent = None

        # How often we restart the local search to find the current incumbent
        self.n_restarts = n_restarts

        super(EnvironmentSearch, self).__init__(acquisition_func, model,
                                                maximize_func, task, save_dir)

#     def env_optimize_posterior_mean_and_std(self, model, X_lower, X_upper,
#                                                is_env, startpoint):
#
#         # We only optimize the posterior in the projected subspace
#         env_values = X_upper[is_env == 1]
#         sub_X_lower = X_lower[is_env == 0]
#         sub_X_upper = X_upper[is_env == 0]
#
#         def f(x):
#             # Project x to the subspace
#             x_ = np.zeros([is_env.shape[0]])
#             x_[is_env == 1] = env_values
#             x_[is_env == 0] = x
#
#             mu, var = model.predict(x_[np.newaxis, :])
#             return (mu + np.sqrt(var))[0, 0]
#             #return mu
#
#         #res = scipy.optimize.basinhopping(f, startpoint[is_env == 0],
#            minimizer_kwargs={"bounds" : zip(sub_X_lower, sub_X_upper),
#            "method": "L-BFGS-B"})
#         res = cma.fmin(f, startpoint[is_env == 0], 0.6,
#                            options={"bounds": [sub_X_lower, sub_X_upper]})
#
#
#         xopt = np.zeros([is_env.shape[0]])
#         xopt[is_env == 1] = env_values
#         #xopt[is_env == 0] = res.x
#         #fval = np.array([res.fun])
#         xopt[is_env == 0] = res[0]
#         fval = np.array([res[1]])
#
#         return xopt, fval

    def env_optimize_posterior_mean_and_std(self, model, X_lower, X_upper,
                                            is_env, startpoint):
        def f(x):
            mu, var = model.predict(x[np.newaxis, :])
            return (mu + np.sqrt(var))[0, 0]

        res = cma.fmin(f, startpoint, 0.6,
                       options={"bounds": [X_lower, X_upper]})

        env_values = X_upper[is_env == 1]
        xopt = res[0]

        # Map incumbent to smax
        xopt[is_env == 1] = env_values

        fval = np.array([res[1]])
        return xopt, fval

    def extrapolative_initial_design(self):

        # Create grid for the system size
        idx = self.task.is_env == 1
        if self.n_init_points == 40:
            grid = [self.task.X_upper[idx] / float(i) for i in [8, 16, 32]]
        else:
            grid = [self.task.X_upper[idx] / float(i) for i in [4, 8, 16, 32]]

        self.time_func_eval = np.zeros([1])
        self.time_overhead = np.zeros([1])
        self.X = np.zeros([1, self.task.n_dims])
        self.Y = np.zeros([1, 1])
        self.Costs = np.zeros([1, 1])

        for i  in range(self.n_init_points):

            for j, s in enumerate(grid):

                start_time = time.time()
                x = np.array([np.random.uniform(self.task.X_lower,
                                                self.task.X_upper,
                                                self.task.n_dims)])
                x[:, self.task.is_env == 1] = s
                overhead = time.time() - start_time

                logging.info("Evaluate: %s" % x)

                start_time = time.time()
                y, c = self.task.evaluate(x)
                time_eval = time.time() - start_time

                if i + j == 0:
                    self.X[i] = x[0, :]
                    self.Y[i] = y[0, :]
                    self.Costs[i] = c[0, :]
                    self.time_overhead[i] = overhead
                    self.time_func_eval[i] = time_eval
                else:
                    self.X = np.append(self.X, x, axis=0)
                    self.Y = np.append(self.Y, y, axis=0)
                    self.Costs = np.append(self.Costs, c, axis=0)

                    self.time_overhead = np.append(self.time_overhead,
                                                   np.array([overhead]),
                                                   axis=0)

                    self.time_func_eval = np.append(self.time_func_eval,
                                                    np.array([time_eval]),
                                                    axis=0)

                # Use best point seen so far as incumbent
                best_idx = np.argmin(self.Y)
                # Copy because we are going to change the system size to smax
                self.incumbent = np.copy(self.X[best_idx])
                self.incumbent_value = self.Y[best_idx]
                bounds_subspace = self.task.X_upper[self.task.is_env == 1]
                self.incumbent[self.task.is_env == 1] = bounds_subspace

                it = i * len(grid) + j
                if self.save_dir is not None and it % self.num_save == 0:

                    self.save_iteration(i * len(grid) + j, costs=self.Costs,
                                        hyperparameters=None,
                                        acquisition_value=0)

        self.n_init_points = self.n_init_points * len(grid)

    def run(self, num_iterations=10, X=None, Y=None, Costs=None):

        self.time_start = time.time()

        if X is None and Y is None:
            # No data yet start with initialization procedure
            # TODO: update to new initial design interface
            self.extrapolative_initial_design()
            #self.initialize()

        else:
            self.X = X
            self.Y = Y
            self.Costs = Costs

        for it in range(self.n_init_points, num_iterations):
            logger.info("Start iteration %d ... ", it)
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

            logger.info("New incumbent %s found in %f seconds",
                        str(self.incumbent), time.time() - start_time_inc)

            # Compute the time we needed to pick a new point
            time_overhead = time.time() - start_time
            self.time_overhead = np.append(self.time_overhead,
                                           np.array([time_overhead]))
            logger.info("Optimization overhead was \
                            %f seconds" % (self.time_overhead[-1]))

            # Evaluate the configuration
            logger.info("Evaluate candidate %s" % (str(new_x)))
            start_time = time.time()
            new_y, new_cost = self.task.evaluate(new_x)
            time_func_eval = time.time() - start_time

            # We model the log costs
            new_cost = np.log(new_cost)

            self.time_func_eval = np.append(self.time_func_eval,
                                            np.array([time_func_eval]))

            logger.info("Configuration achieved a performance \
                    of %f in %s seconds" % (new_y[0, 0], new_cost[0]))

            # Add the new observations to the data
            self.X = np.append(self.X, new_x, axis=0)
            self.Y = np.append(self.Y, new_y, axis=0)
            self.Costs = np.append(self.Costs, new_cost[:, np.newaxis], axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                hypers = self.model.hypers

                self.save_iteration(it, costs=self.Costs[-1],
                                hyperparameters=hypers,
                                acquisition_value=self.acquisition_func(new_x))

        logger.info("Return %s as incumbent" % (str(self.incumbent)))
        return self.incumbent

    def estimate_incumbent(self):

        startpoints = [np.random.uniform(self.task.X_lower,
                                         self.task.X_upper,
                                         self.task.n_dims)
                       for i in range(self.n_restarts)]

        x_opt = np.zeros([len(startpoints), self.task.n_dims])
        fval = np.zeros([len(startpoints)])
        for i, startpoint in enumerate(startpoints):
            logging.info("StartPoint: %s" % startpoint)
            x_opt[i], fval[i] = self.env_optimize_posterior_mean_and_std(
                                                            self.model,
                                                            self.task.X_lower,
                                                            self.task.X_upper,
                                                            self.task.is_env,
                                                            startpoint)

        # Pick best point that was found as incumbent
        best = np.argmin(fval)
        self.incumbent = x_opt[best]
        self.incumbent_value = fval[best]

    def choose_next(self, X=None, Y=None, Costs=None, do_optimize=True):
        if X is not None and Y is not None:
            # Train the model for the objective function and the cost function
            try:
                t = time.time()
                self.model.train(X, Y, do_optimize)
                self.cost_model.train(X, Costs, do_optimize)

                logger.info("Time to train the models: %f", (time.time() - t))
            except Exception:
                logger.error("Model could not be trained with data:", X, Y,
                             Costs)
                raise
            self.model_untrained = False

            # Update the acquisition function with the new models
            self.acquisition_func.update(self.model, self.cost_model)

            # Maximize the acquisition function and return the suggested point
            t = time.time()
            x = self.maximize_func.maximize()
            logger.info("Time to maximize the acquisition function: %f",
                        (time.time() - t))
        else:
            self.initialize()
            x = self.X
        return x
