import time
import logging
import numpy as np


from robo.initial_design.init_random_uniform import init_random_uniform
from robo.solver.base_solver import BaseSolver
from robo.incumbent.best_observation import BestObservation


logger = logging.getLogger(__name__)


class BayesianOptimization(BaseSolver):

    def __init__(self,
            acquisition_func,
            model,
            maximize_func,
            task,
            save_dir=None,
            initial_design=None,
            initial_points=3,
            incumbent_estimation=None,
            num_save=1,
            train_intervall=1,
            n_restarts=1):
        """
        Implementation of the standard Bayesian optimization loop that uses
        an acquisition function and a model to optimize a given task.
        This module keeps track of additional information such as runtime,
        optimization overhead, evaluated points and saves the output
        in a csv file.

        Parameters
        ----------
        acquisition_func: BaseAcquisitionFunctionObject
            The acquisition function which will be maximized.
        model: ModelObject
            Model (i.e. GaussianProcess, RandomForest) that models our current
            believe of the objective function.
        task: TaskObject
            Task object that contains the objective function and additional
            meta information such as the lower and upper bound of the search
            space.
        save_dir: String
            Output path
        initial_design: function
            Function that returns some points which will be evaluated before
            the Bayesian optimization loop is started. This allows to
            initialize the model.
        initial_points: int
            Defines the number of initial points that are evaluated before the
            actual Bayesian optimization.
        incumbent_estimation: IncumbentEstimationObject,
            Object to estimate the incumbent based on the current model. The
            incumbent is the current best guess of the global optimum and is
            estimated in each iteration.
        num_save: int
            Defines after how many iteration the output is saved.
        train_intervall: int
            Specifies after how many iterations the model is retrained.
        n_restarts: int
            How often the incumbent estimation is repeated.
        """

        super(BayesianOptimization, self).__init__(acquisition_func,
                                                    model,
                                                    maximize_func,
                                                    task,
                                                    save_dir)
        self.start_time = time.time()

        if initial_design == None:
            self.initial_design = init_random_uniform
        else:
            self.initial_design = initial_design

        self.X = None
        self.Y = None
        self.time_func_eval = None
        self.time_overhead = None
        self.train_intervall = train_intervall

        self.num_save = num_save
        self.time_start = None

        self.model_untrained = True
        if incumbent_estimation is None:
            self.estimator = BestObservation(self.model,
                                             self.task.X_lower,
                                             self.task.X_upper)
        else:
            self.estimator = incumbent_estimation
        self.incumbent = None
        self.incumbents = []
        self.incumbent_values = []
        self.n_restarts = n_restarts
        self.init_points = initial_points
        self.runtime = []

    def run(self, num_iterations=10, X=None, Y=None):
        """
        The main Bayesian optimization loop

        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        Y: np.ndarray(N,1)
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

        if X is None and Y is None:
            self.time_func_eval = np.zeros([self.init_points])
            self.time_overhead = np.zeros([self.init_points])
            self.X = np.zeros([self.init_points, self.task.n_dims])
            self.Y = np.zeros([self.init_points, 1])

            init = self.initial_design(self.task.X_lower,
                                       self.task.X_upper,
                                       N=self.init_points)

            for i, x in enumerate(init):
                x = x[np.newaxis, :]

                logger.info("Evaluate: %s" % x)

                start_time = time.time()
                y = self.task.evaluate(x)

                self.X[i] = x[0, :]
                self.Y[i] = y[0, :]
                self.time_func_eval[i] = time.time() - start_time
                self.time_overhead[i] = 0.0

                logger.info("Configuration achieved a performance "
                    "of %f in %f seconds" %
                    (self.Y[i], self.time_func_eval[i]))

                # Use best point seen so far as incumbent
                best_idx = np.argmin(self.Y)
                self.incumbent = np.array([self.X[best_idx]])
                self.incumbent_value = np.array([self.Y[best_idx]])

                self.incumbents.append(self.incumbent)
                self.incumbent_values.append(self.incumbent_value)
                self.runtime.append(time.time() - self.start_time)

                if self.save_dir is not None and (i) % self.num_save == 0:
                    self.save_iteration(i, hyperparameters=None,
                                        acquisition_value=0)
                    self.save_json(i)

        else:
            self.X = X
            self.Y = Y
            self.time_func_eval = np.zeros([self.X.shape[0]])
            self.time_overhead = np.zeros([self.X.shape[0]])

#             best = np.argmin(Y)
#             incumbent = X[best]
#             incumbent_value = Y[best]
#             self.incumbents.append(incumbent[np.newaxis, :])
#             self.incumbent_values.append(incumbent_value[np.newaxis, :])
#             self.runtime.append(time.time() - self.start_time)

        for it in range(self.init_points, num_iterations):
            logger.info("Start iteration %d ... ", it)

            start_time = time.time()
            # Choose next point to evaluate
            if it % self.train_intervall == 0:
                do_optimize = True
            else:
                do_optimize = False

            new_x = self.choose_next(self.X, self.Y, do_optimize)

            # Estimate current incumbent
            start_time_inc = time.time()
            startpoints = init_random_uniform(self.task.X_lower,
                                              self.task.X_upper,
                                              self.n_restarts)
            self.incumbent, self.incumbent_value = \
                    self.estimator.estimate_incumbent(startpoints)

            self.incumbents.append(self.incumbent)
            self.incumbent_values.append(self.incumbent_value)

            logger.info("New incumbent %s found in %f seconds with "
                        "estimated performance %f",
                        str(self.incumbent), time.time() - start_time_inc,
                        self.incumbent_value)

            time_overhead = time.time() - start_time
            self.time_overhead = np.append(self.time_overhead,
                                           np.array([time_overhead]))

            logger.info("Optimization overhead was %f seconds" %
                            (self.time_overhead[-1]))

            logger.info("Evaluate candidate %s" % (str(new_x)))
            start_time = time.time()
            new_y = self.task.evaluate(new_x)
            time_func_eval = time.time() - start_time
            self.time_func_eval = np.append(self.time_func_eval,
                                            np.array([time_func_eval]))

            logger.info("Configuration achieved a performance of %f " %
                        (new_y[0, 0]))

            logger.info("Evaluation of this configuration took %f seconds" %
                        (self.time_func_eval[-1]))

            # Update the data
            self.X = np.append(self.X, new_x, axis=0)
            self.Y = np.append(self.Y, new_y, axis=0)

            self.runtime.append(time.time() - self.start_time)

            if self.save_dir is not None and (it) % self.num_save == 0:
                hypers = self.model.hypers
                self.save_iteration(
                    it,
                    hyperparameters=hypers,
                    acquisition_value=self.acquisition_func(new_x))
                self.save_json(it)

        # TODO: Retrain model and then return the incumbent
        logger.info("Return %s as incumbent with predicted performance %f" %
                    (str(self.incumbent), self.incumbent_value))

        return self.incumbent, self.incumbent_value

    def choose_next(self, X=None, Y=None, do_optimize=True):
        """
        Suggests a new point to evaluate.

        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        Y: np.ndarray(N,1)
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

        if X is None and Y is None:
            x = self.initial_design(self.task.X_lower,
                                    self.task.X_upper,
                                    N=1)

        elif X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            x = self.initial_design(self.task.X_lower,
                                    self.task.X_upper,
                                    N=1)
        else:
            try:
                logger.info("Train model...")
                t = time.time()
                self.model.train(X, Y, do_optimize=do_optimize)
                logger.info("Time to train the model: %f", (time.time() - t))
            except:
                logger.error("Model could not be trained", X, Y)
                raise
            self.model_untrained = False
            self.acquisition_func.update(self.model)

            logger.info("Maximize acquisition function...")
            t = time.time()
            x = self.maximize_func.maximize()

            logger.info("Time to maximize the acquisition function: %f", \
                        (time.time() - t))

        return x

    def get_json_data(self, it):
        '''

        Overrides method in base solver.

        '''
        jsonData = dict()
        jsonData = {
                    "optimization_overhead":None if self.time_overhead is None else self.time_overhead[it],
                    "runtime":None if self.time_start is None else time.time() - self.time_start,
                    "incumbent":None if self.incumbent is None else self.incumbent.tolist(),
                    "incumbent_fval":None if self.incumbent_value is None else self.incumbent_value.tolist(),
                    "time_func_eval": self.time_func_eval[it],
                    "iteration":it
                    }
        return jsonData