# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:18:38 2015

@author: aaron
"""

import unittest
import numpy as np

from robo.models.random_forest import RandomForest
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.acquisition.ei import EI
from robo.acquisition.pi import PI
from robo.acquisition.lcb import LCB
from robo.incumbent.best_observation import BestObservation
from robo.incumbent.posterior_optimization import PosteriorMeanOptimization
from robo.incumbent.posterior_optimization import PosteriorMeanAndStdOptimization


class TestRandomForest(unittest.TestCase):

    def test(self):
        X_lower = np.array([0])
        X_upper = np.array([1])
        X = init_random_uniform(X_lower, X_upper, 10)
        Y = np.sin(X)

        model = RandomForest(types=np.zeros([X_lower.shape[0]]))
        model.train(X, Y)

        x_test = init_random_uniform(X_lower, X_upper, 3)

        # Shape matching predict
        m, v = model.predict(x_test)

        assert len(m.shape) == 2
        assert m.shape[0] == x_test.shape[0]
        assert m.shape[1] == 1
        assert len(v.shape) == 2
        assert v.shape[0] == x_test.shape[0]
        assert v.shape[1] == 1

        # Shape matching function sampling
        x_ = np.linspace(X_lower, X_upper, 10)
        x_ = x_[:, np.newaxis]
        #funcs = model.sample_functions(x_, n_funcs=2)
        #assert len(funcs.shape) == 2
        #assert funcs.shape[0] == 2
        #assert funcs.shape[1] == x_.shape[0]

        # Check compatibility with all acquisition functions
        acq_func = EI(model,
                     X_upper=X_upper,
                     X_lower=X_lower)
        acq_func.update(model)
        acq_func(x_test)

        acq_func = PI(model,
                     X_upper=X_upper,
                     X_lower=X_lower)
        acq_func.update(model)
        acq_func(x_test)

        acq_func = LCB(model,
                     X_upper=X_upper,
                     X_lower=X_lower)
        acq_func.update(model)
        acq_func(x_test)

        # Check compatibility with all incumbent estimation methods
        rec = BestObservation(model, X_lower, X_upper)
        inc, inc_val = rec.estimate_incumbent(None)
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == X_upper.shape[0]
        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1

        rec = PosteriorMeanOptimization(model, X_lower, X_upper)
        startpoints = init_random_uniform(X_lower, X_upper, 4)
        inc, inc_val = rec.estimate_incumbent(startpoints)
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == X_upper.shape[0]
        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1

        rec = PosteriorMeanAndStdOptimization(model, X_lower, X_upper)
        startpoints = init_random_uniform(X_lower, X_upper, 4)
        inc, inc_val = rec.estimate_incumbent(startpoints)
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == X_upper.shape[0]
        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1

if __name__ == "__main__":
    unittest.main()
