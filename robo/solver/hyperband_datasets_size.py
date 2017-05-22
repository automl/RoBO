import os
import time
import json
import logging
from robo.solver.base_solver import BaseSolver

import numpy as np

try:
    import multibeep as mb

except ImportError as e:
    raise ValueError("If you want to use Hyperband you have to install the following dependencies:\n"
                     "multibeep (see https://github.com/automl/multibeep)")


logger = logging.getLogger(__name__)


class hyperband_arm(mb.arms.python):
    def __init__(self, task, configuration, subset_fractions, HB_obj):
        self.task = task
        self.configuration = configuration
        self.subset_fractions = subset_fractions
        self.HB_obj = HB_obj
        self.i = 0
        super().__init__(self, b"Hyperband arm wrapper")

    def pull(self):
        if self.i == len(self.subset_fractions):
            raise "Ooops, that shouldn't happen. Trying to pull this arm too many times."
        res = self.task.objective_function(self.configuration,
                                            dataset_fraction=self.subset_fractions[self.i])
        self.i += 1
        self.HB_obj.time_func_eval_SH[-1] += res['cost']

        if res['function_value'] < self.HB_obj.incumbent_values[-1]:
            # append incumbent
            self.HB_obj.incumbents.append(self.configuration)
            self.HB_obj.incumbent_values.append(res['function_value'])

            # compute the total cost for pure function evaluations
            total_cost = sum(self.HB_obj.time_func_eval_SH)
            self.HB_obj.time_func_eval_incumbent.append(total_cost)

            # add runtime info as well
            self.HB_obj.runtime.append(time.time() - self.HB_obj.time_start)

        return -res['function_value']
    # rest of the methods don't have to be specified here


class HyperBand_DataSubsets(BaseSolver):
    """
    variables to use the save_iteration function of the BaseSolver class:


    """
    def __init__(self, task, eta, min_subset_fraction, output_path=None, rng=None):
        """
        Parameters
        ----------

        task : hpolib.benchmark.AbstractBenchmark object
            the task should accept dataset_fraction argument (between 0 and 1).
        eta : float
            In each iteration, a complete run of sequential halving is executed. In it,
            after evaluating each configuration on the same subset size, only a fraction of
            1/eta of them 'advances' to the next round.
            Must be greater or equal to 2.
        min_subset_fraction : float
            size of the smallest subset to consider. The sizes will be
            geometrically distributed $\sim \eta^k$ for
            $k\in [0, 1, ... , num_subsets - 1]$ where
            $\eta^{num_subsets - 1} \geq min_subset_fraction$
        output_path: string
            Specifies the path where the intermediate output after each iteration will be saved.
            If None no output will be saved to disk.
        rng: numpy.random.RandomState
        """

        self.task = task
        self.eta = eta
        self.min_subset_fraction = min_subset_fraction
        self.output_path = output_path

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng
        #TODO: set seed of the configuration space

        task.configuration_space.seed(self.rng.randint(np.iinfo(np.int16).max))

        self.X = []
        self.Y = []

        self.incumbents = [None]
        self.incumbent_values = [np.inf]
        self.time_func_eval_SH = [0]
        self.time_func_eval_incumbent = [0]
        self.runtime = [None]
        self.time_start = None

    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        """
        The main optimization loop

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
        self.time_start = time.time()

        eta = self.eta
        num_subsets = -int(np.log(self.min_subset_fraction)/np.log(eta)) + 1
        subset_fractions = np.power(eta, -np.linspace(num_subsets-1, 0, num_subsets))

        for it in range(num_iterations):

            self.time_func_eval_SH.append(0)

            logger.info("Start iteration %d ... ", it)

            # compute the the value of s for this iteration
            s = num_subsets - 1 - (it % num_subsets)

            # the number of initial configurations
            n = int(np.floor((num_subsets)/(s+1)) * eta**s)

            # set up the arms with random configurations
            configurations = [self.choose_next() for i in range(n)]
            arms = [hyperband_arm( self.task, c,
                                   subset_fractions[(-s-1):], self) for c in configurations]

            # set up the bandit and the policy and play
            bandit = mb.bandits.last_n_pulls(n=1)
            [bandit.add_arm(a) for a in arms]

            policy = mb.policies.successive_halving(
                bandit, 1, eta, factor_pulls = 1)

            policy.play_n_rounds(s+1)

          
            # add all full evaluations to X and Y
            for i in range(len(arms)):
                if arms[bandit[i].identifier].i == bandit[0].num_pulls:
                    self.X.append(arms[bandit[i].identifier].configuration)
                    self.Y.append(bandit[i].estimated_mean)


            if it == 0:
                self.incumbents.pop(0)
                self.incumbent_values.pop(0)
                self.time_func_eval_SH.pop(0)
                self.time_func_eval_incumbent.pop(0)
                self.runtime.pop(0)



            if self.output_path is not None:
                self.save_output(it)

    def choose_next(self, X=None, Y=None):
        """
        Suggests a new point to evaluate.

        Parameters
        ----------
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        Y: np.ndarray(N,1)
            Function values of the already evaluated points

        Returns
        -------
        ConfigSpace.Configuration
            Suggested point
        """
        return self.task.configuration_space.sample_configuration()

    def save_output(self, it):
        data = dict()
        data["runtime"] = self.runtime[it]
        # Note that the ConfigSpace automatically converts to the [0, 1]^D space
        data["incumbent"] = self.incumbents[it].get_array().tolist()
        data["incumbents_value"] = self.incumbent_values[it]
        data["time_func_eval"] = self.time_func_eval[it]
        data["iteration"] = it
        json.dump(data, open(os.path.join(self.output_path, "hyperband_iter_%d.json" % it), "w"))
