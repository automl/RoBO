import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
from models import GPyModel
from acquisition import EI

#TODO: test case for the expected improvement function
class EITestCase1(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[-7.0358, -1.1344]])
        self.y = np.array([[1, 1]])
        self.kernel = GPy.kern.rbf(input_dim=2, variance=12.3, lengthscale=5.0)
        self.model = GPyModel(self.kernel)
        self.model.train(self.x, self.y)

    def test(self):
        # print self.x
        acq_fn = EI(self.model)
        ei_value = acq_fn(self.x)
        print ei_value

if __name__=="__main__":
    unittest.main()
