import unittest
import logging
import numpy as np

from robo.fmin import mtbo

logging.basicConfig(level=logging.INFO)


def objective(x, t):
    if t == 1:
        c = 1000
        y = (x - 0.5) ** 2
    else:
        c = 1
        y = (x - 0.5) ** 2 + 1
    return y[0], c


class TestFminInterface(unittest.TestCase):

    def setUp(self):
        self.lower = np.zeros([2])
        self.upper = np.ones([2]) * 3

    def test_bayesian_optimization(self):
        res = mtbo(objective_function=objective,
                   lower=self.lower,
                   upper=self.upper,
                   n_tasks=2,
                   n_init=2,
                   num_iterations=10)

        assert len(res["x_opt"].shape) == 1
        assert res["x_opt"].shape[0] == self.lower.shape[0]
        assert np.all(res["x_opt"] >= self.lower)
        assert np.all(res["x_opt"] <= self.upper)

if __name__ == "__main__":
    unittest.main()
