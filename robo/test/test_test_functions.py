import random
import unittest
import numpy as np
from robo.benchmarks.test_functions import branin, hartmann6, hartmann3, goldstein_price_fkt

class TestTestFunction(object):
    def test(self):
        ys = self.fkt(self.X_stars)
        assert np.all( ys < (self.Y_star + 0.0001))
        X = np.empty((self.num_t, self.dims))
        Y = np.empty((self.num_t, 1))
        for j in range(self.num_t):
            for i in range(self.dims):
                X[j,i] = random.random() * (self.X_upper[i] - self.X_lower[i]) + self.X_lower[i];
        Y = self.fkt(X)
        print Y.shape, X.shape
        #print Y
        assert np.all(Y >= self.Y_star)
        
    def _setUp(self):
        self.dims = self.fkt.dims
        self.X_lower = self.fkt.X_lower
        self.X_upper = self.fkt.X_upper
        self.X_stars = self.fkt.X_stars
        self.Y_star = self.fkt.Y_star
        
class TestHartmann6Function(unittest.TestCase, TestTestFunction):
    def setUp(self):
        self.fkt = hartmann6
        TestTestFunction._setUp(self)
        self.num_t = 2000

class TestHartmann3Function(unittest.TestCase, TestTestFunction):
    def setUp(self):
        self.fkt = hartmann3
        TestTestFunction._setUp(self)
        self.num_t = 2000

class TestBraninFunction(unittest.TestCase, TestTestFunction):
    def setUp(self):
        self.fkt = branin
        TestTestFunction._setUp(self)
        self.num_t = 2000

        
class TestGoldsteinPriceFkt(unittest.TestCase, TestTestFunction):
    def setUp(self):
        self.fkt = goldstein_price_fkt
        TestTestFunction._setUp(self)
        self.num_t = 2000
    
if __name__=="__main__":
    unittest.main()
