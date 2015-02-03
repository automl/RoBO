import sys
#sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
from robo.models import GPyModel
from robo.acquisition import EI, Entropy
import matplotlib.pyplot as plt

#@unittest.skip("skip first test\n")
class EITestCase1(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[ 0.62971589], [ 0.63273273], [ 0.17867868], [ 0.17447447], [ 1.88558559]]);
        self.y = np.array([[-3.69925653], [-3.66221988], [-3.65560591], [-3.58907791], [-8.06925984]]);
        self.kernel = GPy.kern.RBF(input_dim=1, variance= 30.1646253727, lengthscale = 0.435343653946)    
        self.noise = 1e-10
        self.model = GPyModel(self.kernel, noise_variance=self.noise, optimize=False)
        self.model.train(self.x, self.y)

    def test(self):
        ei_par_0 = EI(self.model, X_upper=np.array([2.1]), X_lower=np.array([-2.1]),  par=0.0, derivative=True)
        ei_par_1 = EI(self.model, X_upper=np.array([2.1]), X_lower=np.array([-2.1]),  par=1.0, derivative=True)
        ei_par_2 = EI(self.model, X_upper=np.array([2.1]), X_lower=np.array([-2.1]),  par=2.0, derivative=True)
        
        out0 = np.array([ ei_par_0(np.array([x]), derivative=True) for x in [[1.0], [0.62971589]]])
        value0 = out0[:,0]
        derivative0 = out0[:,1]
        
        out1 = np.array([ ei_par_1(np.array([x]), derivative=True) for x in [[1.0], [0.62971589]]])
        value1 = out1[:,0]
        derivative1 = out1[:,1]
        
        out2 = np.array([ ei_par_2(np.array([x]), derivative=True) for x in [[1.0], [0.62971589]]])
        value2 = out2[:,0]
        derivative2 = out2[:,1]
        assert(value0[1] <= 1e-5)
        assert(np.all(value0 >= value1))
        assert(np.all(value1 >= value2))
        assert(np.all(derivative0 >= derivative1))
        assert(np.all(derivative1 >= derivative2))


if __name__=="__main__":
    unittest.main()
