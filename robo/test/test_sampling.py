import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
from models import GPyModel
from acquisition import Entropy, PI, EI
from test_functions import branin

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
        self.model = GPyModel(kernel)
        self.model.train(X,Y)

        self.BestGuesses = np.zeros((0, self.D))

    def test(self):
        entropy = Entropy(self.model)
        acquisition_fn = EI(self.model)
        zb, mb = entropy.sample_from_measure(self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        print "zb: ", zb


# @unittest.skip("non empty array, sampling from measure")
class SecondIterationTest(unittest.TestCase):
    def setUp(self):


        self.D = 2 # dimension of input space
        # self.x_prev = np.array([])
        self.xmin = np.array([[-8,-8]])
        self.xmax =  np.array([[19, 19]])
        self.n_representers = 20

        self.X = np.array([[6.8165, 15.1224]])
        self.Y = np.array([[213.3935]])
        self.kernel = GPy.kern.rbf(input_dim = self.D, variance = 13.3440, lengthscale = 4.4958)
        self.model = GPyModel(self.kernel)
        self.model.train(self.X, self.Y)

        self.BestGuesses = np.array([
            [-7.9969,    0.4625],
            [-7.1402,   -7.7652]
        ])

        fac = 42.9076/68.20017903
    # def test(self):
    #     entropy = Entropy(self.model)
    #     # create acquisition function for the sampling of representer points:
    #     acquisition = PI(self.model)
    #     zb, mb = entropy.sample_from_measure(self.GP, self.xmin, self.xmax, 50, self.BestGuesses, acquisition)

    def test(self):
        entropy = Entropy(self.model)
        acquisition_fn = EI(self.model, par = 0)
        zb, mb = entropy.sample_from_measure(self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        print "zb: ", zb

@unittest.skip("third iteration")
class ThirdIterationTest(unittest.TestCase):
    def setUp(self):
        self.D = 2 # dimension of input space
        self.x_prev = np.array([])
        self.xmin = np.array([[-8,-8]])
        self.xmax =  np.array([[19, 19]])
        self.n_representers = 20

        self.X = np.array([
            [6.81648061143168, 15.1224216919405],
            [7.175994953837093, 4.410131392920116]
        ])
        self.Y = np.array([[213.393471408362], [47.849496333858006]])
        self.kernel = GPy.kern.rbf(input_dim = self.D, variance = 3.690198808146339**2,
                                   lengthscale = (1.6963e-11, 4.1267e-36), ARD = True)
        self.model = GPyModel(self.kernel)
        self.model.train(self.X, self.Y)



        self.BestGuesses = np.array([
            [-7.99691187993169,	0.462452810299556],
            [-7.14017599085650,	-7.76524114707603],
            [1.74264535658588, -6.40800531312687]
        ])


    # def test(self):
    #     entropy = Entropy(self.model)
    #     # create acquisition function for the sampling of representer points:
    #     acquisition = PI(self.model)
    #     zb, mb = entropy.sample_from_measure(self.GP, self.xmin, self.xmax, 50, self.BestGuesses, acquisition)

    def test(self):
        entropy = Entropy(self.model)
        acquisition_fn = EI(self.model, par = 0)
        zb, mb = entropy.sample_from_measure(self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        print "zb: ", zb





# TODO: test case for the slice shrink rank sampling method
# class SliceShrinkRankTestCase(unittest.TestCase):


if __name__=="__main__":
    unittest.main()




