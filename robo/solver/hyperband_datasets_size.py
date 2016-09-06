'''
Created on 2016/09/06

@author: Stefan Falkner
'''

import logging
from . import BaseSolver

logger = logging.getLogger(__name__)


class HyperBand_DataSubsets(BaseSolver):

	"""
		variables to use the save_iteration function of the BaseSolver class:
		
			self.X = []
			self.Y = []
			self.incumbent = None
			self.incumbent_value = None
			self.time_func_eval = []
			self.time_overhead = []
			self.time_start = None
	"""
	def __init__ (self, task, factor, min_subset_fraction, budget_per_iteration, time_complexity):
		"""
		Parameters
		----------
		
		task : robo.tasks.BaseTask??
			the task object should interpret the last dimension as the dataset size.
		factor : double
			In each iteration, a complete run of sequential halving is executed. In it,
			after evaluating each configuration on the same subset size, only a fraction of 
			1/factor of them 'advances' to the next round.
			Must be greater or equal to 2.
		
		min_subset_fraction : float
			smallest value for the subset size
		budget_per_iteration : float
			budget per iteration in multiples of function evaluation on the whole data set.
			Must be larger then 1, but should probably be larger than the factor value.
		time_complexity : float
		
		"""


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
        pass

    def choose_next(self, X=None, Y=None):
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

        Returns
        -------
        np.ndarray(1,D)
            Suggested point
        """
        pass
