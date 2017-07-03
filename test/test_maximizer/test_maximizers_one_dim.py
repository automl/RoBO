
import unittest
import numpy as np

from robo.maximizers.direct import Direct
from robo.maximizers.grid_search import GridSearch
from robo.maximizers.random_sampling import RandomSampling


def objective_function(x):
    y = (0.5 - x) ** 2
    return y


class TestMaximizers1D(unittest.TestCase):

    def setUp(self):

        self.lower = np.array([0])
        self.upper = np.array([1])

    def test_direct(self):
        maximizer = Direct(objective_function, self.lower, self.upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert len(x.shape) == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)

    def test_grid_search(self):
        maximizer = GridSearch(objective_function, self.lower, self.upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert len(x.shape) == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)

    def test_random_sampling(self):
        maximizer = RandomSampling(objective_function, self.lower, self.upper)
        x = maximizer.maximize()

        assert x.shape[0] == 1
        assert len(x.shape) == 1
        assert np.all(x >= self.lower)
        assert np.all(x <= self.upper)

if __name__ == "__main__":
    unittest.main()
