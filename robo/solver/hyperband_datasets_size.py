'''
Created on 2016/09/06

@author: Stefan Falkner
'''


import sys
gh_root= '/ihome/sfalkner/repositories/github/'
sys.path.extend([gh_root + 'RoBO/', gh_root + 'HPOlibConfigSpace/'])
sys.path.extend([gh_root + 'HPOlib/package/'])
sys.path.extend([gh_root + 'HPOlib/'])
bb_root= '/ihome/sfalkner/repositories/bitbucket/'
sys.path.extend([gh_root + 'bandits/'])


import time
import logging
from robo.solver.base_solver import BaseSolver

import numpy as np


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
	def __init__ (self, task, eta, num_subsets, save_dir=None, num_save=1, rng=None):
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
		num_subset : int
			number of subsets to consider. The sizes will be distributed geometrically
			$\sim \eta^k$ for $k\in [0, 1, ... , num_subsets - 1]$
		num_save: int
			Defines after how many iteration the output is saved.
			The execution of a Successive Halving run is considered
			a iteration
		save_dir: String
			Output path
		rng: numpy.random.RandomState
		"""


		self.task = task
		self.eta = eta
		self.num_subsets = num_subsets
		self.save_dir = save_dir
		self.num_save = num_save

		if rng is None:
			self.rng = np.random.RandomState(np.random.randint(0, 10000))
		else:
			self.rng = rng


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
		num_subsets = self.num_subsets

		subset_fractions = np.power(eta, -np.linspace(num_subsets-1, 0, num_subsets))

	
		for it in range(num_iterations):
			logger.info("Start iteration %d ... ", it)

			start_time = time.time()

			# compute the the value of s for this iteration
			s = num_subsets - 1 - ( it % (num_subsets) )

			n = int(np.floor( num_subsets/(s+1)))* eta**s

			subsets = subset_fractions[(-s-1):]

			print("="*50)
			print("s=%f"%s)
			print("n=%i"%n)
			print("subsets:{}".format(subsets))

		from IPython import embed
		embed()


			
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

		bounds = np.array(self.task.get_meta_information()['bounds'])

		x = self.rng.uniform(bounds[:,0],bounds[:,1])
		return(x)
