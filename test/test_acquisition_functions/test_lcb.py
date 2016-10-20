import unittest
import numpy as np

from  scipy.optimize import check_grad


from robo.acquisition_functions.lcb import LCB

from ..dummy_model import DemoModel


class TestLCB(unittest.TestCase):

    def setUp(self):
        X = np.random.rand(10, 2)
        y = np.sinc(X * 10 - 5).sum(axis=1)
        self.model = DemoModel()
        self.model.train(X, y)

    def test_compute(self):
        lcb = LCB(self.model)
        print(lcb.compute(np.random.randn(1, 3)))
        print(lcb.compute(np.random.randn(2, 3)))

        print(lcb.compute(np.random.randn(1, 3), derivative=False))
        print(lcb.compute(np.random.randn(2, 3), derivative=False))
#    def setUp(self):
#        self.X_lower = np.array([0])
#        self.X_upper = np.array([1])
#        self.X = init_random_uniform(self.X_lower, self.X_upper, 4)
#        self.Y = np.sin(self.X)
#        self.kernel = GPy.kern.RBF(input_dim=1)

#        self.model = GPyModel(self.kernel)
#        self.model.train(self.X, self.Y)
#        self.lcb = LCB(self.model,
#                       X_upper=self.X_upper,
#                       X_lower=self.X_lower)

#    def test_general_interface(self):

#        X_test = init_random_uniform(self.X_lower, self.X_upper, 10)

#        a, dadx = self.lcb(X_test, True)

#        assert len(a.shape) == 2
#        assert a.shape[0] == X_test.shape[0]
#        assert a.shape[1] == 1
#        assert len(dadx.shape) == 2
#        assert dadx.shape[0] == X_test.shape[0]
#        assert dadx.shape[1] == X_test.shape[1]

#    def test_check_grads(self):
#        x_ = np.array([[np.random.rand()]])

#        assert check_grad(self.lcb, lambda x: self.lcb(x, True)[1], x_) < 1e-5

if __name__ == "__main__":
    unittest.main()
