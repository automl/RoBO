'''
Created on Jun 11, 2015

@author: Aaron Klein
'''

import time
import logging
import numpy as np

from robo.initial_design.init_random_uniform import init_random_uniform
from robo.solver.bayesian_optimization import BayesianOptimization

from enves.extrapolative_initial_design import extrapolative_initial_design
from enves.env_posterior_opt import EnvPosteriorMeanAndStdOptimization

logger = logging.getLogger(__name__)


class Fabolas(BayesianOptimization):

    def __init__(self, acquisition_func,
                 model,
                 cost_model,
                 maximize_func,
                 task,
                 save_dir=None,
                 initialization=None,
                 num_save=1,
                 train_intervall=1,
                 n_restarts=1,
                 incumbent_estimation=None,
                 initial_points=15):
        """
        Fast Bayesian Optimization of Machine Learning Hyperparameters 
        on Large Datasets

        Parameters
        ----------
        acquisition_func : EnvironmentEntropy Object
            The acquisition function to determine the next point to evaluate.
            This object has to be an instance of EnvironmentEntropy class.
        model : Model object
            Models the objective function. The model has to be a
            Gaussian process. If MCMC sampling of the model's hyperparameter is
            performed, make sure that the acquistion_func is of an instance of
            IntegratedAcquisition to marginalise over the GP's hyperparameter.
        cost_model : model
            Models the cost function. The model has to be a Gaussian Process.
        maximize_func : Maximizer object
            Optimizer to maximize the acquisition function.
        task: Task object
            The task that should be optimized. Make sure that it returns the
            function value as well as the cost if the evaluate() function is
            called.
        save_dir : str, optional
            Path where the results file will be saved
        initialization : func, optional
            Initial design function to find some initial points
        num_save : int, optional
            Specifies after how many iterations the results will be written to
            the output file
        train_intervall : int, optional
            Specified after how many iterations the model will be retrained
        n_restarts : int, optional
            How many local searches are performed to estimate the incumbent.
        incumbent_estimation: IncumbentEstimationObject,
            Object to estimate the incumbent based on the current model. The
            incumbent is the current best guess of the global optimum and is
            estimated in each iteration.
        initial_points : int , optional
            How many points are sampled for the initial design

        """
        self.start_time = time.time()
        self.train_intervall = train_intervall
        self.acquisition_func = acquisition_func
        self.model = model
        self.maximize_func = maximize_func
        self.task = task

        self.initialization = initialization
        self.cost_model = cost_model
        self.save_dir = save_dir
        self.num_save = num_save

        if save_dir is not None:
            self.create_save_dir()

        self.X = None
        self.Y = None
        self.C = None
        self.model_untrained = True
        self.incumbent = None
        self.incumbents = []
        self.incumbent_values = []
        self.runtime = []

        # How often we restart the local search to find the current incumbent
        self.n_restarts = n_restarts

        super(Fabolas, self).__init__(acquisition_func, model,
                                                maximize_func, task, save_dir)

        if incumbent_estimation == None:
            self.estimator = EnvPosteriorMeanAndStdOptimization(self.model,
                                                            self.task.X_lower,
                                                            self.task.X_upper,
                                                            self.task.is_env,
                                                            method="cmaes")
        else:
            self.estimator = incumbent_estimation
        self.init_points = initial_points

    def run(self, num_iterations=10, X=None, Y=None, C=None):
        """
        Runs the main Bayesian optimization loop

        Parameters
        ----------
        num_iterations : int, optional
            Specifies the number of iterations.
        X : (N, D) numpy array, optional
            Initial points where BO starts from.
        Y : (N, D) numpy array, optional
            The function values of the initial points. Make sure the number of
            points is the same.
        C : (N, D) numpy array, optional
            The costs of the initial points. Make sure the number of
            points is the same.

        Returns
        -------
        incumbent : (1, D) numpy array
            The estimated optimum that was found after the specified number of
            iterations.
        """
        self.time_start = time.time()

        if X is None and Y is None and C is None:
            self.time_func_eval = np.zeros([1])
            self.time_overhead = np.zeros([1])
            self.X = np.zeros([1, self.task.n_dims])
            self.Y = np.zeros([1, 1])
            self.C = np.zeros([1, 1])

            init = extrapolative_initial_design(self.task.X_lower,
                                       self.task.X_upper,
                                       self.task.is_env,
                                       N=self.init_points)

            for i, x in enumerate(init):
                x = x[np.newaxis, :]
                start_time = time.time()

                logger.info("Evaluate: %s" % x)

                start_time = time.time()
                y, c = self.task.evaluate(x)

                # Transform cost to log scale
                c = np.log(c)

                if i == 0:
                    self.X[i] = x[0, :]
                    self.Y[i] = y[0, :]
                    self.C[i] = c[0, :]
                    self.time_func_eval[i] = time.time() - start_time
                    self.time_overhead[i] = 0.0
                else:
                    self.X = np.append(self.X, x, axis=0)
                    self.Y = np.append(self.Y, y, axis=0)
                    self.C = np.append(self.C, c, axis=0)

                    time_feval = np.array([time.time() - start_time])
                    self.time_func_eval = np.append(self.time_func_eval,
                                                    time_feval, axis=0)
                    self.time_overhead = np.append(self.time_overhead,
                                                   np.array([0]), axis=0)

                logger.info("Configuration achieved a"
                            "performance of %f and %f costs in %f seconds" %
                            (self.Y[i], self.C[i], self.time_func_eval[i]))

                # Use best point seen so far as incumbent
                best_idx = np.argmin(self.Y)
                best_idx = np.argmin(self.Y)
                # Copy because we are going to change the system size to smax
                self.incumbent = np.copy(self.X[best_idx])
                self.incumbent_value = self.Y[best_idx]
                bounds_subspace = self.task.X_upper[self.task.is_env == 1]
                self.incumbent[self.task.is_env == 1] = bounds_subspace

                self.incumbent = self.incumbent[np.newaxis, :]
                self.incumbent_value = self.incumbent_value[np.newaxis, :]

                self.incumbents.append(self.incumbent)
                self.incumbent_values.append(self.incumbent_value)
                self.runtime.append(time.time() - self.start_time)

                if self.save_dir is not None and (i) % self.num_save == 0:
                    self.save_iteration(i, costs=self.C[-1],
                                        hyperparameters=None,
                                        acquisition_value=0)

        else:
            self.X = X
            self.Y = Y
            self.C = C
            self.time_func_eval = np.zeros([self.X.shape[0]])
            self.time_overhead = np.zeros([self.X.shape[0]])

        for it in range(self.init_points, num_iterations):
            logger.info("Start iteration %d ... ", it)
            # Choose a new configuration
            start_time = time.time()
            if it % self.train_intervall == 0:
                do_optimize = True
            else:
                do_optimize = False
            new_x = self.choose_next(self.X, self.Y, self.C, do_optimize)

            # Estimate current incumbent from the posterior
            # over the configuration space
            start_time_inc = time.time()
            startpoints = init_random_uniform(self.task.X_lower,
                                              self.task.X_upper,
                                              self.n_restarts)
            self.incumbent, self.incumbent_value = \
                self.estimator.estimate_incumbent(startpoints)

            self.incumbents.append(self.incumbent)
            self.incumbent_values.append(self.incumbent_value)

            logger.info("New incumbent %s found in %f seconds",
                        str(self.incumbent), time.time() - start_time_inc)

            # Compute the time we needed to pick a new point
            time_overhead = time.time() - start_time
            self.time_overhead = np.append(self.time_overhead,
                                           np.array([time_overhead]))
            logger.info("Optimization overhead was "
                            "%f seconds" % (self.time_overhead[-1]))

            # Evaluate the configuration
            logger.info("Evaluate candidate %s" % (str(new_x)))
            start_time = time.time()
            new_y, new_cost = self.task.evaluate(new_x)
            time_func_eval = time.time() - start_time

            # We model the log costs
            new_cost = np.log(new_cost)

            self.time_func_eval = np.append(self.time_func_eval,
                                            np.array([time_func_eval]))

            logger.info("Configuration achieved a performance "
                    "of %f in %s seconds" % (new_y[0, 0], new_cost[0]))

            # Add the new observations to the data
            self.X = np.append(self.X, new_x, axis=0)
            self.Y = np.append(self.Y, new_y, axis=0)
            self.C = np.append(self.C, new_cost, axis=0)

            self.runtime.append(time.time() - self.start_time)

            if self.save_dir is not None and (it) % self.num_save == 0:
                hypers = self.model.hypers

                self.save_iteration(it, costs=self.C[-1],
                                hyperparameters=hypers,
                                acquisition_value=self.acquisition_func(new_x))

        logger.info("Return %s as incumbent" % (str(self.incumbent)))
        return self.incumbent

    def choose_next(self, X=None, Y=None, C=None, do_optimize=True):
        """
        Performs one single iteration of Bayesian optimization and estimated
        the next point to evaluate.

        Parameters
        ----------
        X : (N, D) numpy array, optional
            The points that have been observed so far. The model is trained on
            this points.
        Y : (N, D) numpy array, optional
            The function values of the observed points. Make sure the number of
            points is the same.
        C : (N, D) numpy array, optional
            The costs of the observed points. Make sure the number of
            points is the same.
        do_optimze : bool, optional
            Specifies if the hyperparamter of the Gaussian process should be
            optimized.

        Returns
        -------
        x : (1, D) numpy array
            The suggested point to evaluate.
        """

        if X is None and Y is None and C is None:
            x = extrapolative_initial_design(self.task.X_lower,
                                       self.task.X_upper,
                                       self.task.is_env,
                                       N=1)
        elif X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            x = extrapolative_initial_design(self.task.X_lower,
                                             self.task.X_upper,
                                             self.task.is_env,
                                             N=1)
        else:
            # Train the model for the objective function and the cost function
            try:
                t = time.time()
                self.model.train(X, Y, do_optimize)
                self.cost_model.train(X, C, do_optimize)

                logger.info("Time to train the models: %f", (time.time() - t))
            except Exception:
                logger.error("Model could not be trained with data:", X, Y, C)
                raise
            self.model_untrained = False

            # Update the acquisition function with the new models
            self.acquisition_func.update(self.model, self.cost_model,
                                         overhead=self.time_overhead[-1])

            # Maximize the acquisition function and return the suggested point
            t = time.time()
            x = self.maximize_func.maximize()
            logger.info("Time to maximize the acquisition function: %f",
                        (time.time() - t))

        return x
