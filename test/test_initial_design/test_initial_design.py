import unittest
import numpy as np

from robo import initial_design


class TestInitialDesign(unittest.TestCase):

    def setUp(self):
        self.n_dim = 4
        self.lower = np.zeros([self.n_dim])
        self.upper = np.ones([self.n_dim])
        self.n_points = 10

    def test_init_random_uniform(self):

        X = initial_design.init_random_uniform(self.lower, self.upper, self.n_points)

        assert X.shape == (self.n_points, self.n_dim)
        assert np.all(np.min(X, axis=0) >= 0)
        assert np.all(np.max(X, axis=0) <= 1)

    def test_init_random_normal(self):
        X = initial_design.init_random_normal(self.lower, self.upper, self.n_points)

        assert X.shape == (self.n_points, self.n_dim)
        assert np.all(np.min(X, axis=0) >= 0)
        assert np.all(np.max(X, axis=0) <= 1)

    def test_init_grid(self):
        X = initial_design.init_grid(self.lower, self.upper, self.n_points)

        assert X.shape == (self.n_points ** self.n_dim, self.n_dim)
        assert np.all(np.min(X, axis=0) >= 0)
        assert np.all(np.max(X, axis=0) <= 1)

    def test_init_latin_hypercube_sampling(self):
        X = initial_design.init_latin_hypercube_sampling(self.lower, self.upper, self.n_points)

        assert X.shape == (self.n_points, self.n_dim)
        assert np.all(np.min(X, axis=0) >= 0)
        assert np.all(np.max(X, axis=0) <= 1)

if __name__ == "__main__":
    unittest.main()
