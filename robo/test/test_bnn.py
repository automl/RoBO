# -*- coding: utf-8 -*-

import unittest
import numpy as np

from robo.models.bnn import SGLDNet
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.acquisition.ei import EI
from robo.acquisition.pi import PI
from robo.acquisition.lcb import LCB
from robo.incumbent.best_observation import BestObservation


class TestSGLDNet(unittest.TestCase):

    def setUp(self):
        self.lower = np.array([0])
        self.upper = np.array([1])
        X = init_random_uniform(self.lower, self.upper, 100)
        Y = np.sin(X)

        self.model = SGLDNet(X.shape[1], X.shape[0])
        self.model.train(X, Y)

    def test_predict(self):
        # Shape matching predict
        x_test = init_random_uniform(self.lower, self.upper, 3)
        m, v = self.model.predict(x_test)

        assert len(m.shape) == 2
        assert m.shape[0] == x_test.shape[0]
        assert m.shape[1] == 1
        assert len(v.shape) == 2
        assert v.shape[0] == x_test.shape[0]
        assert v.shape[1] == 1

    def test_acquisition_functions(self):
        # Check compatibility with all acquisition functions
        x_test = init_random_uniform(self.lower, self.upper, 3)

        acq_func = EI(self.model,
                      X_upper=self.upper,
                      X_lower=self.lower)
        acq_func.update(self.model)
        acq_func(x_test)

        acq_func = PI(self.model,
                      X_upper=self.upper,
                      X_lower=self.lower)
        acq_func.update(self.model)
        acq_func(x_test)

        acq_func = LCB(self.model,
                       X_upper=self.upper,
                       X_lower=self.lower)
        acq_func.update(self.model)
        acq_func(x_test)

    def test_incumbent_estimation(self):
        # Check compatibility with all incumbent estimation methods
        rec = BestObservation(self.model, self.lower, self.upper)
        inc, inc_val = rec.estimate_incumbent(None)
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == self.upper.shape[0]
        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1

if __name__ == "__main__":
    unittest.main()
