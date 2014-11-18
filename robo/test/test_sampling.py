import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
from models import GPyModel
from acquisition import Entropy

# @unittest.skip("empty array, sampling from measure")
class EmptySampleTestCase(unittest.TestCase):
    def setUp(self):
        self.x_prev = np.array([])
        self.xmin = np.array([-8,-8])
        self.xmax =  np.array([19, 19])
        self.n_representers = 50
        # self.kernel = GPy.kern.rbf(input_dim=2, variance=12.3, lengthscale=5.0)
        # self.model = GPyModel(self.kernel)
        self.model = 'model'

        # This is the GP struct from the main loop in the ES algorithm

        # GP              = struct;
        # GP.covfunc      = in.covfunc;
        # GP.covfunc_dx   = in.covfunc_dx;
        # %GP.covfunc_dx   = in.covfunc_dx;
        # %GP.covfunc_dxdz = in.covfunc_dxdz;
        # GP.likfunc      = in.likfunc;
        # GP.hyp          = in.hyp;
        # GP.res          = 1;
        # GP.deriv        = in.with_deriv;
        # GP.poly         = in.poly;
        # GP.log          = in.log;
        # %GP.SampleHypers = in.SampleHypers;
        # %GP.HyperSamples = in.HyperSamples;
        # %GP.HyperPrior   = in.HyperPrior;
        #
        # GP.x            = in.x;
        # GP.y            = in.y;
        # %GP.dy           = in.dy;
        # GP.K            = [];
        # GP.cK           = [];

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
        self.GP['x'] = np.array([])
        self.GP['y'] = np.array([])
        self.D = 2 # dimension of input space


    def test(self):
        entropy = Entropy(self.model)
        zb, mb = entropy.sample_from_measure(self.GP, self.xmin, self.xmax, 50, 0, 0)
        self.assertAlmostEqual(mb[1], -6.59167373)


@unittest.skip("non empty array, sampling from measure")
class NonEmptySampleTestCase(unittest.TestCase):
    def setUp(self):
        self.x_prev = np.array([])
        self.xmin = np.array([-8,-8])
        self.xmax =  np.array([19, 19])
        self.n_representers = 50
        # self.kernel = GPy.kern.rbf(input_dim=2, variance=12.3, lengthscale=5.0)
        # self.model = GPyModel(self.kernel)
        self.model = 'model'

        # This is the GP struct from the main loop in the ES algorithm

        # GP              = struct;
        # GP.covfunc      = in.covfunc;
        # GP.covfunc_dx   = in.covfunc_dx;
        # %GP.covfunc_dx   = in.covfunc_dx;
        # %GP.covfunc_dxdz = in.covfunc_dxdz;
        # GP.likfunc      = in.likfunc;
        # GP.hyp          = in.hyp;
        # GP.res          = 1;
        # GP.deriv        = in.with_deriv;
        # GP.poly         = in.poly;
        # GP.log          = in.log;
        # %GP.SampleHypers = in.SampleHypers;
        # %GP.HyperSamples = in.HyperSamples;
        # %GP.HyperPrior   = in.HyperPrior;
        #
        # GP.x            = in.x;
        # GP.y            = in.y;
        # %GP.dy           = in.dy;
        # GP.K            = [];
        # GP.cK           = [];

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
        self.GP['x'] = np.array([[11.9000, 1.3639], [13.7704, 13.5959]])
        self.GP['y'] = np.array([])
        self.D = 2 # dimension of input space
        self.BestGuesses = np.array([
            [-7.0358, -1.1344],
            [0.0496, 16.7625],
            [7.2164, 18.6712],
            [-7.9998, 18.9999]
        ])


    def test(self):
        entropy = Entropy(self.model)
        zb, mb = entropy.sample_from_measure(self.GP, self.xmin, self.xmax, 50, self.BestGuesses, 0)

# TODO: test case for the slice shrink rank sampling method
class SliceShrinkRankTestCase(unittest.TestCase):


    def test(self):
        # print str(self.xx)
        # print self.xx.shape
        assert(True)

if __name__=="__main__":
    unittest.main()




