import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy

# from models import GPyModel
from acquisition import Entropy, PI, EI
from test_functions import branin

from robo.models import GPyModel
from robo.acquisition import Entropy
from robo.sampling import sample_from_measure

@unittest.skip("empty array, sampling from measure \n")
class FirstIterationTest(unittest.TestCase):
    def setUp(self):

        # set random seed
        np.random.seed(1)

        self.x_prev = np.array([[]])
        self.xmin = np.array([[-8,-8]])
        self.xmax =  np.array([[19, 19]])
        self.n_representers = 20
        # self.kernel = GPy.kern.rbf(input_dim=2, variance=12.3, lengthscale=5.0)
        # self.model = GPyModel(self.kernel)
        self.D = 2 # dimension of input space
        # This is the GP struct from the main loop in the ES algorithm
        #initialize the samples
        X = np.empty((0, self.D))

        Y = np.empty((0, 1))
        #draw a random sample from the objective function in the
        #dimension space
        # X[0,None] = np.random.random() * (self.xmax[0, np.newaxis] - self.xmin[0, np.newaxis]) + self.xmin[0, np.newaxis]
        objective_fkt = branin
        # Y[0] = objective_fkt(X[0,np.newaxis])

        #
        # Building up the model
        #
        kernel = GPy.kern.rbf(input_dim=self.D, variance=12.3, lengthscale=5.0)
        self.model = GPyModel(kernel, optimize = False)
        self.model.train(X,Y)

        self.BestGuesses = np.zeros((0, self.D))

    def test(self):
        entropy = Entropy(self.model)
        acquisition_fn = EI(self.model)
        zb, mb = entropy.sample_from_measure(self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        print "zb: ", zb

@unittest.skip("skipping second iteration, EI\n")
class SecondIterationTestEI(unittest.TestCase):
    def setUp(self):

        # set random seed
        np.random.seed(1)

        self.D = 2 # dimension of input space
        # self.x_prev = np.array([])
        self.xmin = np.array([[-8,-8]])
        self.xmax =  np.array([[19, 19]])
        self.n_representers = 20

        self.X = np.array([[6.8165, 15.1224]])
        self.Y = np.array([[213.3935]])
        self.kernel = GPy.kern.RBF(input_dim = self.D, variance = 13.3440, lengthscale = 4.4958)
        self.model = GPyModel(self.kernel, optimize = False)
        self.model.train(self.X, self.Y)

        self.BestGuesses = np.array([
            [-7.9969,    0.4625],
            [-7.1402,   -7.7652]
        ])
        # fac = 42.9076/68.20017903

    def test(self):
        entropy = Entropy(self.model, self.xmin, self.xmax)
        acquisition_fn = EI(self.model, par = 0, xmin = self.xmin, xmax = self.xmax)
        zb, mb = entropy.sample_from_measure(self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        print "zb: ", zb
        self.assertTrue(True)

@unittest.skip("second iteration, PI")
class SecondIterationTestPI(unittest.TestCase):
    def setUp(self):

        np.random.seed(1)
        self.D = 2 # dimension of input space
        # self.x_prev = np.array([])
        self.xmin = np.array([[-8,-8]])
        self.xmax =  np.array([[19, 19]])
        self.n_representers = 20

        self.X = np.array([[6.8165, 15.1224]])
        self.Y = np.array([[213.3935]])

        self.kernel = GPy.kern.RBF(input_dim = self.D, variance = 13.3440, lengthscale = 4.4958)
        self.model = GPyModel(self.kernel, optimize = False)
        self.model.train(self.X, self.Y)

        self.BestGuesses = np.array([
            [-7.9969,    0.4625],
            [-7.1402,   -7.7652]
        ])
        # fac = 42.9076/68.20017903

    def test(self):
        entropy = Entropy(self.model, self.xmin, self.xmax)
        acquisition_fn = PI(self.model, X_lower = self.xmin, X_upper = self.xmax)
        zb, mb = sample_from_measure(entropy, self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        print "zb: ", zb
        # self.assertTrue(True)

@unittest.skip("test for nullspace projection method")
class ProjNullSpaceTests(unittest.TestCase):
    def setUp(self):
        self.D = 2 # dimension of input space
        # self.x_prev = np.array([])
        self.xmin = np.array([[-8,-8]])
        self.xmax =  np.array([[19, 19]])
        self.n_representers = 20

        self.X = np.array([[6.8165, 15.1224]])
        self.Y = np.array([[213.3935]])
        self.kernel = GPy.kern.rbf(input_dim = self.D, variance = 13.3440, lengthscale = 4.4958)
        self.model = GPyModel(self.kernel, optimize = False)
        self.model.train(self.X, self.Y)

        self.BestGuesses = np.array([
            [-7.9969,    0.4625],
            [-7.1402,   -7.7652]
        ])
        # fac = 42.9076/68.20017903
        self.J = np.zeros((0,0))
        self.v = np.array([[30.7746], [-16.0128]])

        self.JJ = np.array([[0.5135], [-0.8581]])
        self.vv = np.array([[-1.4660], [9.0956]])

    def test(self):
        entropy = Entropy(self.model)
        self.assertEqual(entropy.projNullSpace(self.J, self.v).tolist(),
                         np.array([[30.7746], [-16.0128]]).tolist())
        self.assertEqual(entropy.projNullSpace(self.JJ, self.vv).tolist(),
                         np.array([[2.9283919723599983], [1.7522158685840008]]).tolist())

# @unittest.skip("test for montecarlo sampling method")
class MontecarloSamplerTest(unittest.TestCase):
    def setUp(self):
        self.D = 2 # dimension of input space
        # self.x_prev = np.array([])
        self.xmin = np.array([[-8,-8]])
        self.xmax =  np.array([[19, 19]])
        self.n_representers = 20

        self.X = np.array([[6.8165, 15.1224]])
        self.Y = np.array([[213.3935]])
        self.kernel = GPy.kern.RBF(input_dim = self.D, variance = 13.3440, lengthscale = 4.4958)
        self.model = GPyModel(self.kernel, optimize = False)
        self.model.train(self.X, self.Y)

        self.BestGuesses = np.array([
            [-7.9969,    0.4625],
            [-7.1402,   -7.7652]
        ])
        # fac = 42.9076/68.20017903
        self.J = np.zeros((0,0))
        self.v = np.array([[30.7746], [-16.0128]])

        self.JJ = np.array([[0.5135], [-0.8581]])
        self.vv = np.array([[-1.4660], [9.0956]])

    def test(self):
        entropy = Entropy(self.model, self.xmin, self.xmax, 20)
        entropy.montecarlo_sampler(self.xmin, self.xmax, Nx = 5, Nf = 10)



if __name__=="__main__":
    unittest.main()




