import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
from models import GPyModel
from acquisition import EI

@unittest.skip("skip first test")
class EITestCase1(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[-10, -2]])
        self.xx = np.array([[-90.0358, -1.1344]])
        self.y = np.array([[1, 1]])
        self.kernel = GPy.kern.rbf(input_dim=2, variance=12.3, lengthscale=5.0)
        self.model = GPyModel(self.kernel)
        self.model.train(self.x, self.y)

    def test(self):
        # print self.x
        acq_fn = EI(self.model)
        ei_value = acq_fn(self.xx)
        # print ei_value

# @unittest.skip("skip second test")
class EITestCase2(unittest.TestCase):

    def setUp(self):
        self.dims = 1
        self.X_lower = np.array([-8])
        self.X_upper = np.array([19])
        #initialize the samples
        self.X = np.random.uniform(-3.,3.,(5,1))
        self.Y = np.sin(self.X) + np.random.randn(5,1)*0.05

        #draw a random sample from the objective function in the
        #dimension space
        # X[0,:] = [random.random() * (X_upper[0] - X_lower[0]) + X_lower[0]];
        # objective_fkt= branin2
        # Y[0:] = objective_fkt(X[0,:])

        #
        # Building up the model
        #
        self.kernel = GPy.kern.rbf(input_dim=self.dims, variance=12.3, lengthscale=5.0)
        self.model = GPyModel(self.kernel)
        self.model.train(self.X, self.Y)

    def test(self):
        # print self.model.cK
        acquisition_fn = EI(self.model)
        new_x = np.array([[2.1]])
        print "X: ", self.X
        acquisition_fn(new_x)
        # print self.model.Y
        # self.kernel.dK_dX(np.array([np.ones(len(self.model.X))]), self.model.X)

if __name__=="__main__":
    unittest.main()
