import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
from robo.models import GPyModel
from robo.acquisition import EI
import matplotlib.pyplot as plt

@unittest.skip("skip first test\n")
class EITestCase1(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[-10, -2]])
        self.xx = np.array([[-90.0358, -1.1344]])
        self.y = np.array([[1, 1]])
        self.kernel = GPy.kern.RBF(input_dim=2, variance=12.3, lengthscale=5.0)
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
        self.xmin = np.array([-3])
        self.xmax = np.array([3])
        #initialize the samples
        self.x = np.random.uniform(-3.,3.,(100,1))
        self.y = self.x + 1.13 * np.sin(self.x) - 2.5 * np.cos(3*self.x) + 1.7 * np.cos(17 * self.x)

        # Building up the model
        #
        self.kernel = GPy.kern.RBF(input_dim=self.dims, variance=2, lengthscale=1.5)
        self.model = GPyModel(self.kernel, optimize = False)
        # print "K matrix: ", self.model.K
        self.model.train(self.x, self.y)

    def test(self):
        # print self.model.cK
        acquisition_fn = EI(self.model, self.xmin, self.xmax)

        t = np.arange(-3, 3, 0.1)
        t = t.reshape((t.size, 1))

        y = t + 1.13 * np.sin(t) - 2.5 * np.cos(3*t) + 1.7 * np.cos(17 * t)
        new_y = acquisition_fn(t)

        plt.subplot(211)
        plt.plot(t, y, 'g--')
        plt.plot(self.x, self.y, 'o')

        plt.subplot(212)
        plt.plot(t, new_y, 'r-')
        plt.savefig("test.png")


if __name__=="__main__":
    unittest.main()
