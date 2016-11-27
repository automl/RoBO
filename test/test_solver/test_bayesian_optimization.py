import unittest
import george
import numpy as np

from robo.models.gaussian_process import GaussianProcess
from robo.acquisition_functions.lcb import LCB
from robo.maximizers.direct import Direct
from robo.solver.bayesian_optimization import BayesianOptimization


def objective_func(x):
    y = (x-0.5) ** 2
    return y[0]


class TestBayesianOptimization(unittest.TestCase):

    def setUp(self):
        lower = np.zeros([1])
        upper = np.ones([1])
        kernel = george.kernels.Matern52Kernel(np.array([1]), dim=1, ndim=1)
        model = GaussianProcess(kernel)
        lcb = LCB(model)
        maximizer = Direct(lcb, lower, upper, n_func_evals=10)
        self.solver = BayesianOptimization(objective_func, lower, upper,
                                           lcb, model, maximizer)

    def test_run(self):
        n_iters = 4
        inc, inc_val = self.solver.run(n_iters)

        assert inc.shape[0] == 1
        assert inc >= 0
        assert inc <= 1
        assert len(self.solver.incumbents_values) == n_iters
        assert len(self.solver.incumbents) == n_iters
        assert len(self.solver.time_overhead) == n_iters
        assert len(self.solver.time_func_evals) == n_iters
        assert len(self.solver.runtime) == n_iters
        assert self.solver.X.shape[0] == n_iters
        assert self.solver.y.shape[0] == n_iters

    def test_choose_next(self):
        X = np.random.rand(10, 1)
        y = np.array([objective_func(x) for x in X])
        x_new = self.solver.choose_next(X, y)
        assert x_new.shape[0] == 1
        assert x_new >= 0
        assert x_new <= 1
