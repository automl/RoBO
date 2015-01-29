import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
from robo.models import GPyModel
from robo.acquisition import EI, Entropy
import matplotlib.pyplot as plt

#@unittest.skip("skip first test\n")
class EITestCase1(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[ 0.49207017], [ 0.49399399], [-0.42252252], [ 1.80570571], [ 1.8015015 ]])
        self.y = np.array([[-5.04055516], [-5.02800794], [ 8.56648054], [-9.34341391], [-9.38323318]])
        #self.xx = np.array([[0.49399399], [-1], [2.0]])
        self.kernel = GPy.kern.RBF(input_dim=1, variance= 38.2441777589, lengthscale = 0.369019360085)
        self.kernel = GPy.kern.RBF(input_dim=1, variance= 46.7009706712, lengthscale = 0.414883807937)
        
        self.model = GPyModel(self.kernel, noise_variance=1e-10, optimize=False)
        print self.model.m
        self.model.train(self.x, self.y)

    def test(self):
        # print self.x
        #Entropy()
        entropy = Entropy(self.model, X_upper=np.array([2.1]), X_lower=np.array([-2.1]))
        entropy.update(self.model)
        acq_fn = entropy.sampling_acquisition
        #acq_fn = EI(self.model,  X_upper=[2.1], X_lower=[-2.1], par=0.1)
        xx = np.linspace(-2.1,2.1, 1000)[:,np.newaxis]
        ei_value = acq_fn(xx, verbose=True, derivatives=True)
        print self.model.m
        print ei_value

@unittest.skip("skip second test")
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
