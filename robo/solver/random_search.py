
import time
import numpy as np
import logging

from robo.solver.base_solver import BaseSolver
from robo.incumbent.best_observation import BestObservation

logger = logging.getLogger(__name__)


class RandomSearch(BaseSolver):

    def __init__(self, task=None, save_dir=None, num_save=1, rng=None):
        """
        Random Search [1] that simply evaluates random points. We do not have
        any priors thus we sample points uniformly at random.

        [1] J. Bergstra and Y. Bengio.
            Random search for hyper-parameter optimization.
            JMLR, 2012.

        Parameters
        ----------
        task: TaskObject
            Task object that contains the objective function and additional
            meta information such as the lower and upper bound of the search
            space.
        num_save: int
            Defines after how many iteration the output is saved.
        save_dir: String
            Output path
        rng: numpy.random.RandomState

        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.task = task
        self.save_dir = save_dir

        self.X = None
        self.Y = None

        self.estimator = BestObservation(self,
                                         self.task.X_lower,
                                         self.task.X_upper)
        self.time_func_eval = None
        self.time_overhead = None

        self.num_save = num_save

        self.model_untrained = True

        self.incumbent = None
        self.incumbents = []
        self.incumbent_values = []
        self.runtime = []
        if self.save_dir is not None:
            self.create_save_dir()

    def run(self, num_iterations=10):
        """
        The main optimization loop

        Parameters
        ----------
        num_iterations: int
            The number of iterations

        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        self.time_start = time.time()

        for it in range(num_iterations):
            logger.info("Start iteration %d ... ", it)

            start_time = time.time()
            # Choose next point to evaluate

            new_x = self.choose_next()

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

            self.runtime.append(time.time() - self.time_start)

            # Update the data
            if self.X is None and self.Y is None:
                self.X = new_x
                self.Y = new_y
            else:
                self.X = np.append(self.X, new_x, axis=0)
                self.Y = np.append(self.Y, new_y, axis=0)

            # The incumbent is just the best observation we have seen so far
            start_time_inc = time.time()

            self.incumbent, self.incumbent_value = \
                    self.estimator.estimate_incumbent(None)

            self.incumbents.append(self.incumbent)
            self.incumbent_values.append(self.incumbent_value)

            logger.info("New incumbent %s found in %f seconds with "
                        "estimated performance %f",
                        str(self.incumbent), time.time() - start_time_inc,
                        self.incumbent_value)

            if self.save_dir is not None and (it) % self.num_save == 0:
                self.save_iteration(it)

        logger.info("Return %s as incumbent with predicted performance %f" %
                    (str(self.incumbent), self.incumbent_value))

        return self.incumbent, self.incumbent_value

    def choose_next(self):
        """
        Sample a new point uniformly at random.

        Returns
        -------
        np.ndarray(1,D)
            Suggested point to evaluate
        """
        x = self.rng.uniform(self.task.X_lower,
                                 self.task.X_upper)
        if type(x) == np.float:
            return np.array([[x]])
        else:                                 
            return x[np.newaxis, :]
