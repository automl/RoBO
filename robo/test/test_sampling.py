import sys, os
sys.path.insert(0, '../')
import unittest
import numpy as np
import GPy
import matplotlib.pyplot as plt
import pylab
from datetime import datetime
# from models import GPyModel
from acquisition import Entropy, PI, EI, LogEI
from test_functions import branin

from robo.models import GPyModel
from robo.acquisition import Entropy
from robo.sampling import sample_from_measure

import emcee
here = os.path.abspath(os.path.dirname(__file__))
@unittest.skip("empty array, sampling from measure \n")
class SampleFromSin(unittest.TestCase):
    
    def _some_obj(self, x, invertsign=False, derivative=False):
        if np.any(x < self.xmin) or np.any(x > self.xmax):
             return 0, 0 if derivative else 0
        return (np.sin(x) + 1, np.cos(x)) if derivative else np.sin(x) + 1
    def setUp(self):
        pylab.ion()
        self.objective = self._some_obj
        self.xmin = np.array([0])
        self.xmax = np.array([2*np.pi])
        self.n_representers = 10000
        self.BestGuesses =  np.zeros((0, 1))
        self.X = np.array([[1],[2]])
        
    def test1(self):
        self.fig = plt.figure()
        zb, mb = sample_from_measure(self, self.xmin, self.xmax, self.n_representers, self.BestGuesses, self.objective)
        
        dest_folder =here+"/tmp/test_entopy/vis"
        plt.hist(zb, bins=30, normed=True)
        plt.show()
        self.fig.savefig(dest_folder + "/" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S")+ "sampling.png", format='png')
        self.fig.clf()
        plt.close()

class SampleFromEI(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[ 0.62971589], [ 0.63273273], [ 0.17867868], [ 0.17447447], [ 1.88558559]]);
        self.y = np.array([[-3.69925653], [-3.66221988], [-3.65560591], [-3.58907791], [-8.06925984]]);
        self.kernel = GPy.kern.RBF(input_dim=1, variance= 30.1646253727, lengthscale = 0.435343653946)    
        self.BestGuesses =  np.zeros((0, 1))
        self.noise = 1e-4
        self.model = GPyModel(self.kernel, noise_variance=self.noise, optimize=False)
        self.model.train(self.x, self.y)
        self.n_representers = 100
        self.plotting_range = np.linspace(-2.1,2.1, num=1000)
  
    def test1(self):      
        return 
        self.xmin = np.array([-2.1])
        self.xmax = np.array([2.1])
        acquisition_fn = EI(self.model,self.xmin, self.xmax )
        self.fig = plt.figure()
        zb, mb = sample_from_measure(self.model, self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        dest_folder =here+"/tmp/test_entopy/vis"
        
        ax = self.fig.add_subplot(3, 1, 1)
        ax.hist(zb, bins=30, normed=False)
        
        acq_v =  np.array([ acquisition_fn(np.array([x]))[0] for x in self.plotting_range[:,np.newaxis] ])
        
        ax = self.fig.add_subplot(3, 1, 2)
        ax.plot(zb[:, 0],  np.exp(mb[:, 0]), color="red", marker="*")
        
        ax.plot(self.plotting_range, acq_v)
        ax = self.fig.add_subplot(3, 1, 3)
        ax.plot(self.plotting_range, np.clip(np.log(acq_v), -10, 5))
        self.fig.savefig(dest_folder + "/" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S")+ "sampling.png", format='png')
        self.fig.clf()
        plt.show()
        
    def test2(self):      
        self.xmin = np.array([-2.1])
        self.xmax = np.array([2.1])
        #acquisition_fn = LogEI(self.model,self.xmin, self.xmax )
        acquisition_fn = LogEI(self.model,self.xmin, self.xmax )
        def acquisition_fn_wrapper(x):
            return acquisition_fn(np.array([x]))[0]
        self.fig = plt.figure()
        nwalkers = 10000
        dim = 1
        restarts = np.zeros((nwalkers, dim))    
        restarts[0:nwalkers, ] = self.xmin+ (self.xmax-self.xmin)* np.random.uniform( size = (nwalkers, dim))
        sampler = emcee.EnsembleSampler(nwalkers, 1, acquisition_fn_wrapper)
        zb, lmb, _ = sampler.run_mcmc(restarts, 20)
        print zb
        #zb, mb = sample_from_measure(self.model, self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        dest_folder =here+"/tmp/test_entopy/vis"
        
        ax = self.fig.add_subplot(3, 1, 1)
        ax.hist(zb, bins=30, normed=False)
        
        acq_v =  np.array([ acquisition_fn(np.array([x]))[0] for x in self.plotting_range[:,np.newaxis] ])
        
        ax = self.fig.add_subplot(3, 1, 2)
        #ax.plot(zb[:, 0],  np.exp(mb[:, 0]), color="red", marker="*")
        
        ax.plot(self.plotting_range, acq_v)
        ax = self.fig.add_subplot(3, 1, 3)
        ax.plot(self.plotting_range, np.clip(np.log(acq_v), -10, 5))
        self.fig.savefig(dest_folder + "/" + datetime.now().strftime("%Y.%m.%d.%H.%M.%S")+ "_emcee.png", format='png')
        self.fig.clf()
        plt.show() 
      
# @unittest.skip("empty array, sampling from measure \n")
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

# @unittest.skip("skipping second iteration, EI\n")
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

# @unittest.skip("second iteration, PI")
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

    def test(self):
        entropy = Entropy(self.model, self.xmin, self.xmax)
        acquisition_fn = PI(self.model, X_lower = self.xmin, X_upper = self.xmax)
        zb, mb = sample_from_measure(entropy, self.xmin, self.xmax, self.n_representers, self.BestGuesses, acquisition_fn)
        print "zb: ", zb
        # self.assertTrue(True)

# @unittest.skip("test for nullspace projection method")
class ProjNullSpaceTests(unittest.TestCase):

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
        from robo.sampling import projNullSpace
        entropy = Entropy(self.model, self.xmin, self.xmax)
        self.assertEqual(projNullSpace(self.J, self.v).tolist(),
                         np.array([[30.7746], [-16.0128]]).tolist())
        self.assertEqual(projNullSpace(self.JJ, self.vv).tolist(),
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
        from robo.sampling import montecarlo_sampler
        entropy = Entropy(self.model, self.xmin, self.xmax, 20)
        montecarlo_sampler(self.xmin, self.xmax, Nx = 5, Nf = 10)



if __name__=="__main__":
    unittest.main()




