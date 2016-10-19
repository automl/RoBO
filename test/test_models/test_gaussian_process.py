import george
import unittest
import numpy as np

from robo.models.gaussian_process import GaussianProcess
from robo.priors.default_priors import TophatPrior
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.acquisition.ei import EI
from robo.acquisition.pi import PI
from robo.acquisition.lcb import LCB
from robo.acquisition.information_gain import InformationGain
from robo.incumbent.best_observation import BestObservation
from robo.incumbent.posterior_optimization import PosteriorMeanOptimization
from robo.incumbent.posterior_optimization import PosteriorMeanAndStdOptimization


class TestGaussianProcess(unittest.TestCase):

    def setUp(self):
        X = np.random.rand(10, 2)
        y = np.sinc(X * 10 - 5).sum(axis=1)

        kernel = george.kernels.Matern52Kernel(np.ones(X.shape[1]),
                                               ndim=X.shape[1])

        prior = TophatPrior(-2, 2)
        self.model = GaussianProcess(kernel, prior=prior)
        self.model.train(X, y, do_optimize=False)

    def test_predict(self):

        X_test = np.random.rand(10, 2)

        # Shape matching predict
        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

        m, v = self.model.predict(X_test, full_cov=True)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 2
        assert v.shape[0] == X_test.shape[0]
        assert v.shape[1] == X_test.shape[0]

    def test_sample_function(self):
        pass
        # Shape matching function sampling
        # x_ = np.linspace(X_lower, X_upper, 10)
        # x_ = x_[:, np.newaxis]
        # funcs = model.sample_functions(x_, n_funcs=2)
        # assert len(funcs.shape) == 2
        # assert funcs.shape[0] == 2
        # assert funcs.shape[1] == x_.shape[0]
        #
        # # Shape matching predict variance
        # x_test1 = np.array([np.random.rand(1)])
        # x_test2 = np.random.rand(10)[:, np.newaxis]
        # var = model.predict_variance(x_test1, x_test2)
        # assert len(var.shape) == 2
        # assert var.shape[0] == x_test2.shape[0]
        # assert var.shape[1] == 1
        #
        # # Check compatibility with all acquisition functions
        # acq_func = EI(model,
        #               X_upper=X_upper,
        #               X_lower=X_lower)
        # acq_func.update(model)
        # acq_func(x_test)
        #
        # acq_func = PI(model,
        #               X_upper=X_upper,
        #               X_lower=X_lower)
        # acq_func.update(model)
        # acq_func(x_test)
        #
        # acq_func = LCB(model,
        #                X_upper=X_upper,
        #                X_lower=X_lower)
        # acq_func.update(model)
        # acq_func(x_test)
        #
        # acq_func = InformationGain(model,
        #                            X_upper=X_upper,
        #                            X_lower=X_lower)
        # acq_func.update(model)
        # acq_func(x_test)
        # # Check compatibility with all incumbent estimation methods
        # rec = BestObservation(model, X_lower, X_upper)
        # inc, inc_val = rec.estimate_incumbent(None)
        # assert len(inc.shape) == 2
        # assert inc.shape[0] == 1
        # assert inc.shape[1] == X_upper.shape[0]
        # assert len(inc_val.shape) == 2
        # assert inc_val.shape[0] == 1
        # assert inc_val.shape[1] == 1
        #
        # rec = PosteriorMeanOptimization(model, X_lower, X_upper)
        # startpoints = init_random_uniform(X_lower, X_upper, 4)
        # inc, inc_val = rec.estimate_incumbent(startpoints)
        # assert len(inc.shape) == 2
        # assert inc.shape[0] == 1
        # assert inc.shape[1] == X_upper.shape[0]
        # assert len(inc_val.shape) == 2
        # assert inc_val.shape[0] == 1
        # assert inc_val.shape[1] == 1
        #
        # rec = PosteriorMeanAndStdOptimization(model, X_lower, X_upper)
        # startpoints = init_random_uniform(X_lower, X_upper, 4)
        # inc, inc_val = rec.estimate_incumbent(startpoints)
        # assert len(inc.shape) == 2
        # assert inc.shape[0] == 1
        # assert inc.shape[1] == X_upper.shape[0]
        # assert len(inc_val.shape) == 2
        # assert inc_val.shape[0] == 1
        # assert inc_val.shape[1] == 1

if __name__ == "__main__":
    unittest.main()
