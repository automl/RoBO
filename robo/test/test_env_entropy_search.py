'''
Created on Jul 1, 2015

@author: Aaron Klein
'''
import setup_logger
import logging
import unittest
import numpy as np

import GPy
from robo.models.gpy_model import GPyModel
from robo.acquisition.environment_entropy import EnvironmentEntropy
from robo.recommendation.incumbent import compute_incumbent


logger = logging.getLogger(__name__)


class TestEnvEntropySearch(unittest.TestCase):

    def setUp(self):
        n_points = 10
        n_feat = 3
        self.X_lower = np.zeros([n_feat])
        self.X_upper = np.ones([n_feat])
        self.X = np.random.randn(n_points, n_feat)
        self.y = np.random.randn(n_points)[:, np.newaxis]
        self.kernel = GPy.kern.RBF(input_dim=n_feat)
        self.model = GPyModel(self.kernel, optimize=False)
        self.model.train(self.X, self.y)
        self.is_env = np.array([0, 0, 1])
        self.env_es = EnvironmentEntropy(self.model, self.model, self.X_lower,
                                         self.X_upper,
                                         compute_incumbent,
                                         self.is_env)

    def test_sample_representer_points(self):
        # Check if representers are in the correct subspace
        self.env_es.update_representer_points()

        reps = self.env_es.zb

        assert np.all(reps[:, 2] == self.X_upper[2])

    def test_check_gradients(self):
        pass

    def test_check_interface(self):
        pass

if __name__ == "__main__":
    unittest.main()
