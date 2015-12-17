# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:08:04 2015

@author: aaron
"""


import time
import logging
import numpy as np

import cma

from robo.solver.bayesian_optimization import BayesianOptimization


logger = logging.getLogger(__name__)


class MultiTaskBO(BayesianOptimization):

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
                 n_init_points=15):
        """
        Solver class that implements MultiTaskBO by Swersky et al.

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
        task: MultiTask object
            The task that should be optimzed. Make sure that it returns the
            function value as well as the cost if the evaluate() function is
            called.
        save_dir : str, optional
            Path where the results file will be saved
        initialization : func, optional
            Initial design function to find some intial points
        num_save : int, optional
            Specifies after how many iterations the results will be written to
            the output file
        train_intervall : int, optional
            Specified after how many iterations the model will be retrained
        n_restarts : int, optional
            How many local searches are performed to estimate the incumbent.
        n_init_points : int , optional
            How many points are sampled for the intial design

        """
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

        super(MultiTaskBO, self).__init__(acquisition_func, model,
                                                maximize_func, task, save_dir)


    def run(self, num_iterations=10, X=None, Y=None, Costs=None):
        """
        Runs the main Bayesian optimization loop

        Parameters
        ----------
        num_iterations : int, optional
            Specifies the number of iterations.
        X : (N, D) numpy array, optional
            Initial points where BO starts from.
        Y : (N, D) numpy array, optional
            The function values of the intial points. Make sure the number of
            points is the same.
        Costs : (N, D) numpy array, optional
            The costs of the intial points. Make sure the number of
            points is the same.

        Returns
        -------
        incumbent : (1, D) numpy array
            The estimated optimium that was found after the specified number of
            iterations.
        """
        self.time_start = time.time()

        if X is None and Y is None:
            # No data yet start with initialization procedure
            # TODO: update to new initial design interface
            self.initialize()

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
            self._estimate_incumbent()

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
            self.Costs = np.append(self.Costs, new_cost, axis=0)

            if self.save_dir is not None and (it) % self.num_save == 0:
                hypers = self.model.hypers

                self.save_iteration(it, costs=self.Costs[-1],
                                    hyperparameters=hypers,
                                    acquisition_value=self.acquisition_func(new_x))

        logger.info("Return %s as incumbent" % (str(self.incumbent)))
        return self.incumbent

    def _estimate_incumbent(self):

        startpoints = [np.random.uniform(self.task.X_lower,
                                         self.task.X_upper,
                                         self.task.n_dims)
                       for i in range(self.n_restarts)]

        x_opt = np.zeros([len(startpoints), self.task.n_dims])
        fval = np.zeros([len(startpoints)])
        for i, startpoint in enumerate(startpoints):
          
        #TODO Implement optimization strategy
          
        # Pick best point that was found as incumbent
        best = np.argmin(fval)
        self.incumbent = x_opt[best]
        self.incumbent_value = fval[best]

    def choose_next(self, X=None, Y=None, Costs=None, do_optimize=True):
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
        Costs : (N, D) numpy array, optional
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
