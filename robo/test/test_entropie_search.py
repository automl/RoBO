import os
import random
import errno
import unittest
import matplotlib; #matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import GPy
import pylab as pb 
import numpy as np
#pb.ion()
from robo.models import GPyModel
from robo.test_functions import branin2, branin
from robo.acquisition import PI, UCB, Entropy, EI
from robo.maximize import cma, DIRECT, grid_search
here = os.path.abspath(os.path.dirname(__file__))

#
# Plotting Stuff.
# It's really ugly until now, because it only fitts to 1D and to the branin function
# where the second domension is 12
#

obj_samples = 700
plot_min = -8
plot_max = 19
plotting_range = np.linspace(plot_min, plot_max, num=obj_samples)
second_branin_arg = np.empty((obj_samples,))
second_branin_arg.fill(12)
branin_arg = np.append(np.reshape(plotting_range, (obj_samples, 1)), np.reshape(second_branin_arg, (obj_samples, 1)), axis=1)
branin_result = branin(branin_arg)
branin_result = branin_result.reshape(branin_result.shape[0],)

def _plot_model(model, acquisition_fkt, objective_fkt, i, callback):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    model.m.plot(ax=ax, plot_limits=[plot_min, plot_max])
    xlim_min, xlim_max, ylim_min, ylim_max =  ax.axis()
    ax.set_ylim(min(np.min(branin_result), ylim_min), max(np.max(branin_result), ylim_max))
    c1 = np.reshape(plotting_range, (obj_samples, 1))
    c2 = acquisition_fkt(c1)
    c2 = c2*50 / np.max(c2)
    c2 = np.reshape(c2,(obj_samples,))
    ax.plot(plotting_range,c2, 'r')
    ax.plot(plotting_range, branin_result, 'black')
    callback(ax)
    fig.savefig("%s/../tmp/pmin_%s.png"%(here, i), format='png')
    fig.clf()
    plt.close()
    
class EmptySampleTestCase(unittest.TestCase):
    def setUp(self):

        dims = 1
        self.num_initial_vals = 3
        self.X_lower = np.array([-8]);#, -8])
        self.X_upper = np.array([19]);#, 19])
        #initialize the samples
        X = np.empty((self.num_initial_vals, dims))
    
        Y = np.empty((self.num_initial_vals, 1))
        self.x_values = [12.0, 4.0, 8.0, 3.0, -8.0, -4.0, 19.0, 0.1, 16.0, 15.6]
        #draw a random sample from the objective function in the
        #dimension space 
        self.objective_fkt= branin2
        for j in range(self.num_initial_vals):
            for i in range(dims):
                X[j,i] = self.x_values[j]#random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
            Y[j:] = self.objective_fkt( np.array([X[j,:]]))
        
        #
        # Building up the model
        #    
        #old gpy version 
        try:
            self.kernel = GPy.kern.rbf(input_dim=dims, variance=28.5375**2, lengthscale=5.059)
        #gpy version >=0.6
        except AttributeError, e:
            self.kernel = GPy.kern.RBF(input_dim=dims, variance=28.5375**2, lengthscale=5.059)
            
        self.model = GPyModel(self.kernel, optimize=False, noise_variance =0.0015146**2)
        self.model.train(X,Y)
        #self.model.m.optimize()
        #
        # creating an acquisition function
        #
        self.acquisition_fkt = Entropy(self.model, self.X_lower, self.X_upper)
    @unittest.skip("skip it")
    def test_pmin(self):
        for i in xrange(self.num_initial_vals, len(self.x_values)):
            self.acquisition_fkt.update(model)
            new_x = np.array(self.x_values[i]).reshape((1,1,))#  grid_search(self.acquisition_fkt, self.X_lower, self.X_upper)
            new_y = self.objective_fkt(new_x)
            def plot_pmin(ax):
                ax.plot(self.acquisition_fkt.zb,np.exp(self.acquisition_fkt.logP)*100, marker="o", color="#ff00ff", linestyle="");
                ax.plot(self.acquisition_fkt.zb,self.acquisition_fkt.logP, marker="h", color="#00a0ff", linestyle="");
                ax.text(0.95, 0.01, str(self.acquisition_fkt.current_entropy),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='#222222', fontsize=10)
            _plot_model(self.model, self.acquisition_fkt, self.objective_fkt, i, plot_pmin)
            self.model.update(np.array(new_x), np.array(new_y))
            
    def test_innovation(self):
        self.acquisition_fkt = Entropy(self.model, self.X_lower, self.X_upper, Nb = 10)
        Var= np.array( [[  493.2476,  192.0936,  189.3891,  426.5045,  419.9972,  332.1155,  483.1472,  -30.6904,  437.6728,  475.2670], 
                        [  192.0936,  267.2359,  265.4794,   88.2276,  288.7455,   45.6296,  146.3654, -112.2569,  278.3833,  242.1045], 
                        [  189.3891,  265.4794,  263.7466,   86.6537,  285.6440,   44.7031,  144.0873, -112.2385,  275.2662,  239.0783], 
                        [  426.5045,   88.2276,   86.6537,  498.2159,  274.9279,  469.3545,  471.2575,   -9.0593,  297.7650,  360.6824], 
                        [  419.9972,  288.7455,  285.6440,  274.9279,  460.5593,  179.1422,  368.1959,  -69.6378,  463.5500,  455.9222], 
                        [  332.1155,   45.6296,   44.7031,  469.3545,  179.1422,  498.5377,  396.1277,   -3.5359,  198.9042,  258.4307], 
                        [  483.1472,  146.3654,  144.0873,  471.2575,  368.1959,  396.1277,  496.6195,  -19.5925,  389.6205,  441.9061], 
                        [  -30.6904, -112.2569, -112.2385,   -9.0593,  -69.6378,   -3.5359,  -19.5925,  132.7377,  -63.3793,  -46.8737], 
                        [  437.6728,  278.3833,  275.2662,  297.7650,  463.5500,  198.9042,  389.6205,  -63.3793,  468.7849,  467.3753], 
                        [  475.2670,  242.1045,  239.0783,  360.6824,  455.9222,  258.4307,  441.9061,  -46.8737,  467.3753,  484.3670]])
        zb  = np.array([ [0.1569],  [6.4306],  [6.4735], [-2.6102],  [2.7876], [-4.3457], [-0.9570], [13.6003],  [2.4161],  [1.3773]])
        lmb = np.array([ 4.1938,  3.1827,  3.1675,  4.2753,  3.9806,  4.2928,  4.2381,  2.4608,  4.0232,  4.1183])
        Mb  = np.array([ 7.6510, 50.3527, 50.7310,  2.0226, 20.4149,  0.7512,  4.6460, 63.3195, 18.0728, 12.4844])

        self.acquisition_fkt.update(model);
        #mu, var = self.model.predict(np.array(zb), full_cov=True)
        logP,dlogPdMu,dlogPdSigma,dlogPdMudMu = self.acquisition_fkt._joint_min(Mb, Var, with_derivatives=True)

        L = self.acquisition_fkt._get_gp_innovation_local(zb);
        L(np.array([[2.0]]))
        
if __name__=="__main__":
    unittest.main()
