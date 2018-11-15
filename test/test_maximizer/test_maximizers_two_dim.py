
import unittest
import numpy as np

from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.acquisition_functions.base_acquisition import BaseAcquisitionFunction
from test.dummy_model import DemoQuadraticModel


class DemoAcquisitionFunction(BaseAcquisitionFunction):

    def __init__(self):
        model = DemoQuadraticModel()
        X = np.random.rand(10, 2)
        y = X ** 2
        model.train(X, y[:, 0])
        super(DemoAcquisitionFunction, self).__init__(model)

    def compute(self, x, **kwargs):
        y = np.sum((0.5 - x) ** 2, axis=1)
        return np.array([y])


class TestMaximizers2D(unittest.TestCase):

    def setUp(self):

        self.lower = np.array([0, 0])
        self.upper = np.array([1, 1])
        self.objective_function = DemoAcquisitionFunction()

    def test_random_sampling(self):
        maximizer = RandomSampling(self.objective_function, self.lower, self.upper)
        x = maximizer.maximize()

        assert x.shape[0] == 2
        assert len(x.shape) == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)

    def test_differential_evolution(self):
        maximizer = DifferentialEvolution(self.objective_function, self.lower, self.upper, n_iters=10)
        x = maximizer.maximize()

        assert x.shape[0] == 2
        assert len(x.shape) == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)

    def test_scipy(self):
        maximizer = SciPyOptimizer(self.objective_function, self.lower, self.upper)
        x = maximizer.maximize()

        assert x.shape[0] == 2
        assert len(x.shape) == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)


if __name__ == "__main__":
    unittest.main()
