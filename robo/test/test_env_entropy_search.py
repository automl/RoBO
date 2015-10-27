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
from robo.acquisition.log_ei import LogEI
from robo.acquisition.environment_entropy import EnvEntropySearch
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
        self.env_es = EnvEntropySearch(self.model, self.model, self.X_lower, self.X_upper, compute_incumbent, self.is_env)

    def test_loss_kl_div(self):

        # Test Case 1: KL-Loss between uniform base measure and direct delta
        pmin = np.array([0, 0, 0, 1, 0, 0])
        log_proposal_vals = np.ones([pmin.shape[0]]) / float(pmin.shape[0])
        kl_div = self.env_es._loss_kl_div(log_proposal_vals, pmin)
        assert kl_div == 0.0

        # Test Case 2: KL-Loss between uniform base measure and uniform base measure
        pmin = np.ones([pmin.shape[0]]) / float(pmin.shape[0])
        log_proposal_vals = np.ones([pmin.shape[0]]) / float(pmin.shape[0])
        kl_div_base = self.env_es._loss_kl_div(log_proposal_vals, pmin)
        assert kl_div_base > 0.0

        # Test Case 3: KL-Loss between uniform base measure and a normal-like distributions
        pmin = np.array([0, 0.1, 0.2, 0.4, 0.2, 0.1, 0.0])
        log_proposal_vals = np.ones([pmin.shape[0]]) / float(pmin.shape[0])
        kl_div = self.env_es._loss_kl_div(log_proposal_vals, pmin)
        assert kl_div < kl_div_base and kl_div > 0.0

    def test_sample_representer_points(self):
        # Sample some representer points and check if they are in the correct subspace
        proposal_measure = LogEI(self.model, self.X_lower, self.X_upper, compute_incumbent)
        reps = self.env_es._sample_representers(proposal_measure, n_representers=10, n_dim=self.X_lower.shape[0])
        assert np.all(reps[:, 2] == self.X_upper[2])

    def test_compute_pmin(self):
        proposal_measure = LogEI(self.model, self.X_lower, self.X_upper, compute_incumbent)
        reps = self.env_es._sample_representers(proposal_measure, n_representers=10, n_dim=self.X_lower.shape[0])
        pmin = self.env_es._compute_pmin(self.model, reps, num_func_samples=100)
        logger.info(pmin)
        assert pmin.shape[0] == reps.shape[0]

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
