import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
from models import GPyModel
from acquisition import Entropy, PI, EI
from test_functions import branin

# @unittest.skip("empty array, sampling from measure")
class EmptySampleTestCase(unittest.TestCase):
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


@unittest.skip("non empty array, sampling from measure")
class NonEmptySampleTestCase(unittest.TestCase):
    def setUp(self):
        self.x_prev = np.array([])
        self.xmin = np.array([-8])
        self.xmax =  np.array([19])
        self.n_representers = 50
        # self.BestGuesses = np.array([np.zeros(1)])

        self.X = np.random.uniform(-3.,3.,(20,1))
        self.Y = np.sin(self.X) + np.random.randn(20,1)*0.05
        self.kernel = GPy.kern.rbf(input_dim=1, variance=12.3, lengthscale=5.0)
        self.model = GPyModel(self.kernel)
        self.model.train(self.X, self.Y)


        # This is the GP struct from the main loop in the ES algorithm

        self.GP = {}
        self.GP['covfunc'] = lambda x: x
        self.GP['covfunc_dx'] = lambda x: x
        self.GP['covfunc_dxdz'] = lambda x: x
        self.GP['likfunc'] = lambda x: x
        self.GP['hyp'] = 'hyperparameters'
        self.GP['res'] = 1
        self.GP['deriv'] = False # with_derivative - this is supplied by the user as an argument to ES
        self.GP['poly'] = -1 # polnomial mean?
        self.GP['log'] = 0 # logarithmic transformed observations?
        self.GP['x'] = np.array([[11.9000], [13.7704]])
        self.GP['y'] = np.array([])
        self.D = 2 # dimension of input space
        self.BestGuesses = np.array([
            [-7.0358],
            [0.0496 ],
            [7.2164 ],
            [-7.9998]
        ])


    # def test(self):
    #     entropy = Entropy(self.model)
    #     # create acquisition function for the sampling of representer points:
    #     acquisition = PI(self.model)
    #     zb, mb = entropy.sample_from_measure(self.GP, self.xmin, self.xmax, 50, self.BestGuesses, acquisition)

    def test(self):
        import GPy
        from models import GPyModel
        from test_functions import branin
        kernel = GPy.kern.rbf(input_dim=2, variance=6.5816*6.5816, lengthscale=[5.9076, 5.9076], ARD=True)
        X_lower = np.array([-8,-8])
        X_upper = np.array([19, 19])
        X = np.empty((1, 2))
        Y = np.empty((1, 1))
        X[0,:] = [2.6190,    5.4830] #random.random() * (X_upper[0] - X_lower[0]) + X_lower[0]];

        objective_fkt = branin

        Y[0:] = objective_fkt(X)
        model = GPyModel(kernel,noise_variance=0.044855)
        model.train(X,Y)
        model.m.optimize()
        acquisition_fn = EI(self.model)
        entropy = Entropy(self.model)
        # zb, mb = entropy.sample_from_measure(self.GP, self.xmin, self.xmax, 50, 0, 0)
        zb, mb = entropy.sample_from_measure(self.GP, self.xmin, self.xmax, 50, self.BestGuesses, acquisition_fn)
        print "zb: ", zb
        print "mb: ", mb


# TODO: test case for the slice shrink rank sampling method
# class SliceShrinkRankTestCase(unittest.TestCase):


if __name__=="__main__":
    unittest.main()




