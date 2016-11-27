import unittest
import numpy as np

from robo.fmin import random_search, bayesian_optimization


def objective(x):
    y = (x - 0.5) ** 2
    return y


class TestFminInterface(unittest.TestCase):

    def setUp(self):
        self.lower = np.zeros([1])
        self.upper = np.ones([1])

    def test_random_search(self):
        res = random_search(objective_function=objective, lower=self.lower, upper=self.upper)
        assert len(res["x_opt"].shape) == 1
        assert res["x_opt"].shape[0] == 1
        assert res["x_opt"] >= 0
        assert res["x_opt"] <= 1

    def test_bayesian_optimization(self):
        res = bayesian_optimization(objective_function=objective,
                                    lower=self.lower,
                                    upper=self.upper,
                                    num_iterations=10)
        assert len(res["x_opt"].shape) == 1
        assert res["x_opt"].shape[0] == 1
        assert res["x_opt"] >= 0
        assert res["x_opt"] <= 1

if __name__ == "__main__":
    unittest.main()
