import os
import time
import json
import logging
import numpy as np

from robo.initial_design.init_random_uniform import init_random_uniform
from robo.solver.base_solver import BaseSolver


logger = logging.getLogger(__name__)


class BayesianOptimization(BaseSolver):

    def __init__(self, objective_func, lower, upper,
                 acquisition_func, model, maximize_func,
                 initial_design=init_random_uniform,
                 initial_points=3,
                 output_path=None,
                 train_interval=1,
                 n_restarts=1,
                 rng=None):
        """
        Implementation of the standard Bayesian optimization loop that uses
        an acquisition function and a model to optimize a given objective_func.
        This module keeps track of additional information such as runtime,
        optimization overhead, evaluated points and saves the output
        in a json file.

        Parameters
        ----------
        acquisition_func: BaseAcquisitionFunctionObject
            The acquisition function which will be maximized.
        model: ModelObject
            Model (i.e. GaussianProcess, RandomForest) that models our current
            believe of the objective function.
        objective_func:
            Function handle for the objective function
        output_path: string
            Specifies the path where the intermediate output after each iteration will be saved.
            If None no output will be saved to disk.
        initial_design: function
            Function that returns some points which will be evaluated before
            the Bayesian optimization loop is started. This allows to
            initialize the model.
        initial_points: int
            Defines the number of initial points that are evaluated before the
            actual Bayesian optimization.
        train_interval: int
            Specifies after how many iterations the model is retrained.
        n_restarts: int
            How often the incumbent estimation is repeated.
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(100000))
        else:
            self.rng = rng

        self.model = model
        self.acquisition_func = acquisition_func
        self.maximize_func = maximize_func
        self.start_time = time.time()
        self.initial_design = initial_design
        self.objective_func = objective_func
        self.X = None
        self.y = None
        self.time_func_evals = []
        self.time_overhead = []
        self.train_interval = train_interval
        self.lower = lower
        self.upper = upper
        self.output_path = output_path
        self.time_start = None

        self.incumbents = []
        self.incumbents_values = []
        self.n_restarts = n_restarts
        self.init_points = initial_points
        self.runtime = []

    def run(self, num_iterations=10, X=None, y=None):
        """
        The main Bayesian optimization loop

        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,1)
            Function values of the already evaluated points

        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        # Save the time where we start the Bayesian optimization procedure
        self.time_start = time.time()

        if X is None and y is None:

            # Initial design
            X = []
            y = []

            start_time_overhead = time.time()
            init = self.initial_design(self.lower,
                                       self.upper,
                                       self.init_points,
                                       rng=self.rng)
            time_overhead = (time.time() - start_time_overhead) / self.init_points

            for i, x in enumerate(init):

                logger.info("Evaluate: %s", x)

                start_time = time.time()
                new_y = self.objective_func(x)

                X.append(x)
                y.append(new_y)
                self.time_func_evals.append(time.time() - start_time)
                self.time_overhead.append(time_overhead)

                logger.info("Configuration achieved a performance of %f in %f seconds",
                            y[i], self.time_func_evals[i])

                # Use best point seen so far as incumbent
                best_idx = np.argmin(y)
                incumbent = X[best_idx]
                incumbent_value = y[best_idx]

                self.incumbents.append(incumbent.tolist())
                self.incumbents_values.append(incumbent_value)

                self.runtime.append(time.time() - self.start_time)

                if self.output_path is not None:
                    self.save_output(i)

            self.X = np.array(X)
            self.y = np.array(y)
        else:
            self.X = X
            self.y = y

        # Main Bayesian optimization loop
        for it in range(self.init_points, num_iterations):
            logger.info("Start iteration %d ... ", it)

            start_time = time.time()

            if it % self.train_interval == 0:
                do_optimize = True
            else:
                do_optimize = False

            # Choose next point to evaluate
            new_x = self.choose_next(self.X, self.y, do_optimize)

            self.time_overhead.append(time.time() - start_time)
            logger.info("Optimization overhead was %f seconds", self.time_overhead[-1])
            logger.info("Next candidate %s", str(new_x))

            # Evaluate
            start_time = time.time()
            new_y = self.objective_func(new_x)
            self.time_func_evals.append(time.time() - start_time)

            logger.info("Configuration achieved a performance of %f ", new_y)
            logger.info("Evaluation of this configuration took %f seconds", self.time_func_evals[-1])

            # Extend the data
            self.X = np.append(self.X, new_x[None, :], axis=0)
            self.y = np.append(self.y, new_y)

            # Estimate incumbent
            best_idx = np.argmin(self.y)
            incumbent = self.X[best_idx]
            incumbent_value = self.y[best_idx]

            self.incumbents.append(incumbent.tolist())
            self.incumbents_values.append(incumbent_value)
            logger.info("Current incumbent %s with estimated performance %f",
                        str(incumbent), incumbent_value)

            self.runtime.append(time.time() - self.start_time)

            if self.output_path is not None:
                self.save_output(it)

        logger.info("Return %s as incumbent with error %f ",
                    self.incumbents[-1], self.incumbents_values[-1])

        return self.incumbents[-1], self.incumbents_values[-1]

    def choose_next(self, X=None, y=None, do_optimize=True):
        """
        Suggests a new point to evaluate.

        Parameters
        ----------
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,1)
            Function values of the already evaluated points
        do_optimize: bool
            If true the hyperparameters of the model are
            optimized before the acquisition function is
            maximized.
        Returns
        -------
        np.ndarray(1,D)
            Suggested point
        """

        if X is None and y is None:
            x = self.initial_design(self.lower, self.upper, 1, rng=self.rng)[0, :]

        elif X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            x = self.initial_design(self.lower, self.upper, 1, rng=self.rng)[0, :]

        else:
            try:
                logger.info("Train model...")
                t = time.time()
                self.model.train(X, y, do_optimize=do_optimize)
                logger.info("Time to train the model: %f", (time.time() - t))
            except:
                logger.error("Model could not be trained!")
                raise
            self.acquisition_func.update(self.model)

            logger.info("Maximize acquisition function...")
            t = time.time()
            x = self.maximize_func.maximize()

            logger.info("Time to maximize the acquisition function: %f", (time.time() - t))

        return x

    def save_output(self, it):

        data = dict()
        data["optimization_overhead"] = self.time_overhead[it]
        data["runtime"] = self.runtime[it]
        data["incumbent"] = self.incumbents[it]
        data["incumbents_value"] = self.incumbents_values[it]
        data["time_func_eval"] = self.time_func_evals[it]
        data["iteration"] = it

        json.dump(data, open(os.path.join(self.output_path, "robo_iter_%d.json" % it), "w"))

