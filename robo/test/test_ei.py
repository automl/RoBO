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
        self.X = np.random.uniform(-3.,3.,(100,1))
        self.Y = np.sin(self.X) + np.random.randn(100,1)*0.05

        # Building up the model
        #
        self.kernel = GPy.kern.RBF(input_dim=self.dims, variance=2, lengthscale=1.5)
        self.model = GPyModel(self.kernel, optimize = False)
        # print "K matrix: ", self.model.K
        self.model.train(self.X, self.Y)

    def test(self):
        # print self.model.cK
        acquisition_fn = EI(self.model, self.xmin, self.xmax)
        new_x = np.array([[2.1]])

        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.X, self.Y, 'bo')
        plt.savefig("test.png")
        # plt.subplot(212)
        # plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
        # plt.show()

        print "X: ", self.X
        print new_x

        new_y = acquisition_fn(new_x)
        print new_y
        # print self.model.Y
        # self.kernel.dK_dX(np.array([np.ones(len(self.model.X))]), self.model.X)

if __name__=="__main__":
    unittest.main()
