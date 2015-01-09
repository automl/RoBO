import random
import unittest
import numpy as np
from robo.test_functions import branin, hartmann6, hartmann3, goldstein_price_fkt

class TestTestFunction(object):
    def test(self):
        ys = self.fkt(self.X_star)
        assert np.all( ys < (self.Y_star + 0.0001))
        X = np.empty((self.num_t, self.dims))
        Y = np.empty((self.num_t, 1))
        for j in range(self.num_t):
            for i in range(self.dims):
                X[j,i] = random.random() * (self.X_upper[i] - self.X_lower[i]) + self.X_lower[i];
        Y = self.fkt(X)
        assert np.all(Y >= self.Y_star)

class TestHartmann6Function(unittest.TestCase, TestTestFunction):
    def setUp(self):
        self.X_lower = np.array([0.0,0.0,0.0,0.0,0.0,0.0]);
        self.X_upper = np.array([1.0,1.0,1.0,1.0,1.0,1.0]);
        self.X_star =  np.array(((0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),))
        self.Y_star = -3.32237
        self.fkt = hartmann6
        self.dims = 6
        self.num_t = 2000

class TestHartmann3Function(unittest.TestCase, TestTestFunction):
    def setUp(self):
        self.X_lower = np.array([0.0,0.0,0.0]);
        self.X_upper = np.array([1.0,1.0,1.0]);
        self.X_star =  np.array(((0.114614, 0.555649, 0.852547),))
        self.Y_star = -3.86278
        self.fkt = hartmann3
        self.dims = 3
        self.num_t = 2000

class TestBraninFunction(unittest.TestCase, TestTestFunction):
    def setUp(self):
        self.X_lower = np.array([-5, 0]);
        self.X_upper = np.array([10, 15]);
        self.X_star =  np.array(((-np.pi, 12.275),(np.pi, 2.275), (9.42478, 2.475)))
        self.Y_star = 0.397887
        self.fkt = branin
        self.dims = 2
        self.num_t = 2000
        
class TestGoldsteinPriceFkt(unittest.TestCase, TestTestFunction):
    def setUp(self):
        self.X_lower = np.array([-2, -2]);
        self.X_upper = np.array([2, 2]);
        self.X_star =  np.array(((0,-1), ))
        self.Y_star = 3
        self.fkt = goldstein_price_fkt
        self.dims = 2
        self.num_t = 2000
    
if __name__=="__main__":
    unittest.main()
