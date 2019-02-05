import unittest

import numpy as np

from robo.fmin import random_search, bayesian_optimization, entropy_search


def objective(x):
    y = (x - 0.5) ** 2
    return y[0]


class TestFminInterface(unittest.TestCase):
    def setUp(self):
        self.lower = np.zeros([1])
        self.upper = np.ones([1])

    def test_random_search(self):
        res = random_search(objective_function=objective,
                            lower=self.lower, upper=self.upper,
                            num_iterations=3)
        assert len(res["x_opt"]) == 1
        assert np.array(res["x_opt"]) >= 0
        assert np.array(res["x_opt"]) <= 1

    def test_entropy_search(self):
        res = entropy_search(objective_function=objective,
                             lower=self.lower, upper=self.upper,
                             n_init=2,
                             num_iterations=3)
        assert len(res["x_opt"]) == 1
        assert np.array(res["x_opt"]) >= 0
        assert np.array(res["x_opt"]) <= 1

    def test_bayesian_optimization_gp_mcmc(self):
        res = bayesian_optimization(objective_function=objective,
                                    lower=self.lower,
                                    upper=self.upper,
                                    n_init=2,
                                    num_iterations=3)
        assert len(res["x_opt"]) == 1
        assert np.array(res["x_opt"]) >= 0
        assert np.array(res["x_opt"]) <= 1

    def test_bohamiann(self):
        res = bayesian_optimization(objective_function=objective,
                                    lower=self.lower,
                                    upper=self.upper,
                                    n_init=2,
                                    num_iterations=3,
                                    model_type="bohamiann")
        assert len(res["x_opt"]) == 1
        assert np.array(res["x_opt"]) >= 0
        assert np.array(res["x_opt"]) <= 1

    def test_dngo(self):
        res = bayesian_optimization(objective_function=objective,
                                    lower=self.lower,
                                    upper=self.upper,
                                    n_init=2,
                                    num_iterations=3,
                                    model_type="dngo")
        assert len(res["x_opt"]) == 1
        assert np.array(res["x_opt"]) >= 0
        assert np.array(res["x_opt"]) <= 1

    def test_gp(self):
        res = bayesian_optimization(objective_function=objective,
                                    lower=self.lower,
                                    upper=self.upper,
                                    n_init=2,
                                    num_iterations=3,
                                    model_type="gp")
        assert len(res["x_opt"]) == 1
        assert np.array(res["x_opt"]) >= 0
        assert np.array(res["x_opt"]) <= 1

    def test_rf(self):
        res = bayesian_optimization(objective_function=objective,
                                    lower=self.lower,
                                    upper=self.upper,
                                    n_init=2,
                                    num_iterations=3,
                                    model_type="rf")
        assert len(res["x_opt"]) == 1
        assert np.array(res["x_opt"]) >= 0
        assert np.array(res["x_opt"]) <= 1


if __name__ == "__main__":
    unittest.main()
