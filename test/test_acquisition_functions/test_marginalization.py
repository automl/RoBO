import unittest
import numpy as np

from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.pi import PI
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.models.base_model import BaseModel


class DemoModel(BaseModel):

    def __init__(self, m, v):
        self.m = m
        self.v = v

    def train(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        m = np.ones(X_test.shape[0]) * self.m
        v = np.ones(X_test.shape[0]) * self.v
        return m, v


class DemoModelMCMC(BaseModel):

    def __init__(self):
        self.n_hypers = 10
        self.models = [DemoModel(np.random.randn(), np.random.rand()) for _ in range(self.n_hypers)]

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
        m = 0
        v = 0
        for model in self.models:
            m_, v_ = model.predict(X_test)
            m += m_
            v += v_

        return m / self.n_hypers, v / self.n_hypers

    @BaseModel._check_shapes_train
    def train(self, X, y):
        for model in self.models:
            model.train(X, y)


class TestMarginalizationGPMCMC(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = DemoModelMCMC()
        self.model.train(self.X, self.y)

    def test_lcb(self):
        lcb = LCB(self.model)
        acq = MarginalizationGPMCMC(lcb)

        X_test = np.random.rand(5, 2)
        a = acq.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

    def test_ei(self):
        ei = EI(self.model)
        acq = MarginalizationGPMCMC(ei)

        X_test = np.random.rand(5, 2)
        a = acq.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

    def test_log_ei(self):
        log_ei = LogEI(self.model)
        acq = MarginalizationGPMCMC(log_ei)

        X_test = np.random.rand(5, 2)
        a = acq.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

    def test_pi(self):
        pi = PI(self.model)
        acq = MarginalizationGPMCMC(pi)

        X_test = np.random.rand(5, 2)
        a = acq.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1


if __name__ == "__main__":
    unittest.main()
