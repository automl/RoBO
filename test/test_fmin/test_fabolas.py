import unittest
import numpy as np

from robo.fmin import fabolas


def objective(x, s):

    s_min = 10
    s_max = 10000
    s_ = (np.log(s) - np.log(s_min)) / (np.log(s_max) - np.log(s_min))
    c = s_
    y = (x - 0.5) ** 2 + 1
    y *= (1 - s_ + 1e-4) ** 2
    return y[0], c


class TestFminInterfaceFabolas(unittest.TestCase):

    def setUp(self):
        self.lower = np.zeros([2])
        self.upper = np.ones([2]) * 3

    def test_bayesian_optimization(self):
        res = fabolas(objective_function=objective,
                      lower=self.lower,
                      upper=self.upper,
                      subsets=[10, 20],
                      s_min=10,
                      s_max=10000,
                      n_init=2,
                      num_iterations=3)

        assert len(res["x_opt"]) == self.lower.shape[0]
        assert np.all(np.array(res["x_opt"]) >= self.lower)
        assert np.all(np.array(res["x_opt"]) <= self.upper)

if __name__ == "__main__":
    unittest.main()
