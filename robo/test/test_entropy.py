import random
random.seed(12)
import sys
import os
#sys.path.insert(0, '../')
import unittest
import errno
import numpy as np
np.random.seed(12)
import GPy
import scipy
import robo
from robo.models import GPyModel
from robo.acquisition import EI, Entropy
from robo.visualization import Visualization
import matplotlib.pyplot as plt
from datetime import datetime
class Dummy(object):
    pass

here = os.path.abspath(os.path.dirname(__file__))

@unittest.skip("skip first test\n")
class EntropyTestCase1(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[-1.01216433], [ 0.58432685], [ 0.54567976], [ 0.31943842], [ 0.98867407], [ 0.77694305]])
        self.y = np.array([[ 0.8340836 ], [-4.22582812], [-4.61606884], [-5.16633511], [-0.02364996], [-1.7958453 ]])
        self.kernel = GPy.kern.RBF(input_dim = 1, variance = 7.08235794307, lengthscale = 0.367927189668)    
        self.noise = 1e-3
        self.model = GPyModel(self.kernel, noise_variance=self.noise, optimize=False)
        self.model.train(self.x, self.y)
        #self.bigger_noise = 1e2
        #self.noise_model = GPyModel(self.kernel, noise_variance=self.bigger_noise, optimize=False)
        #self.noise_model.train(self.x, self.y)

    def test(self):
        X_upper = np.array([ 2.1])
        X_lower = np.array([-2.1])
        entropy = Entropy(self.model, X_upper=X_upper, X_lower=X_lower,  derivative=True)
        x_values = [0.62971589, 0.82971589, 0.9,1.1, 0.31944842]
        #x_values = [0.31944842] 
        entropy.update(self.model)
        scipy.io.savemat(here+'/../../../entropie_search/EntropySearch/test.mat', dict(zb = entropy.zb,
                                                    lmb = entropy.lmb, 
                                                    logP = entropy.logP,
                                                    dlogPdM = entropy.dlogPdMu,
                                                    dlogPdV = entropy.dlogPdSigma,
                                                    ddlogPdMdM = entropy.dlogPdMudMu,
                                                    W = entropy.W))
        out0 = np.array([ entropy(np.array([[x]]), derivative=True) for x in x_values])
        value0 = out0[:,0]
        
        
        print out0
        bo_dummy = Dummy()
        bo_dummy.X_upper = X_upper
        bo_dummy.X_lower = X_lower
        bo_dummy.acquisition_fkt = entropy
        bo_dummy.dims = 1
        bo_dummy.model = self.model
        bo_dummy.acquisition_fkt.update( bo_dummy.model)
        dest_folder =here+"/tmp/test_entopy/vis"
        if dest_folder is not None:
            try:
                os.makedirs(dest_folder)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
        Visualization(bo_dummy, dest_folder=dest_folder, prefix=datetime.now().strftime("%Y.%m.%d.%H.%M.%S"), new_x = None, X=self.x, Y=self.y, acq_method = True, obj_method = False, model_method = True, )

@unittest.skip("skip second test\n")
class EntopyTestCase2(unittest.TestCase):
    def setUp(self):
        self.bo, new_x, self.x, self.y = robo.BayesianOptimization.from_iteration("/home/kaiserj/tmp/entropy_robo_example3", 8)
        self.model = self.bo.model
        #print bo.model.m, "\n",X,"\n", Y
    def test(self):
        X_upper = np.array([ 2.1])
        X_lower = np.array([-2.1])
        entropy = Entropy(self.model, X_upper=X_upper, X_lower=X_lower,  derivative=True)
        x_values = [-1.2, -1.19, -1.18]
        #x_values = [0.31944842] 
        entropy.update(self.model)
        scipy.io.savemat(here+'/../../../entropie_search/EntropySearch/test.mat', dict(zb = entropy.zb,
                                                    lmb = entropy.lmb, 
                                                    logP = entropy.logP,
                                                    dlogPdM = entropy.dlogPdMu,
                                                    dlogPdV = entropy.dlogPdSigma,
                                                    ddlogPdMdM = entropy.dlogPdMudMu,
                                                    W = entropy.W))
        
        print "updated"
        out0 = np.array([ entropy(np.array([[x]]), derivative=True) for x in x_values])
        value0 = out0[:,0]
        dest_folder =here+"/tmp/test_entopy/vis"
        if dest_folder is not None:
            try:
                os.makedirs(dest_folder)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
        self.bo.acquisition_fkt = entropy
        Visualization(self.bo, dest_folder=dest_folder, prefix=datetime.now().strftime("%Y.%m.%d.%H.%M.%S"), new_x = None, X=self.x, Y=self.y, acq_method = True, obj_method = False, model_method = True, )
        print out0
        

class EntopyTestCase3(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[-1.01216433], [ 0.58432685], [ 0.54567976], [ 0.31943842], [ 0.98867407], [ 0.77694305]])
        self.y = np.array([[ 0.8340836 ], [-4.22582812], [-4.61606884], [-5.16633511], [-0.02364996], [-1.7958453 ]])
        self.kernel = GPy.kern.RBF(input_dim = 1, variance = 7.08235794307, lengthscale = 0.367927189668)    
        self.noise = 1e-1
        self.model = GPyModel(self.kernel, noise_variance=self.noise, optimize=False)
        self.model.train(self.x, self.y)
        self.X_upper = np.array([ 2.1])
        self.X_lower = np.array([-2.1])
        
    def test(self):
        X_upper = np.array([ 2.1])
        X_lower = np.array([-2.1])
        entropy = Entropy(self.model, X_upper=X_upper, X_lower=X_lower,  derivative=True)
        entropy.model = self.model
        
        #entropy.zb, entropy.lmb = robo.sampling.sample_from_measure(self.model, self.X_lower, self.X_upper, 100, entropy.BestGuesses, entropy.sampling_acquisition)
        """entropy.update_representer_points();
        print entropy.sampling_acquisition(np.array([[0.0]]))
        mu, var = self.model.predict(np.array(entropy.zb), full_cov=True)
        
        scipy.io.savemat(here+'/../../../entropie_search/EntropySearch/samples.mat', dict(zb = entropy.zb,
                                            lmb = entropy.lmb, 
                                            Mb = mu[:,None], 
                                            Vb = var
                                            ))
        
        print "sampled"
        logP, dlogPdMu, dlogPdSigma, dlogPdMudMu = entropy._joint_min(mu, var, with_derivatives=True)"""
        dict_b = scipy.io.loadmat(here+'/../../../entropie_search/EntropySearch/samples.mat')
        entropy.zb = dict_b["zb"]
        entropy.lmb = dict_b["lmb"]
        dict_P = scipy.io.loadmat(here+'/../../../entropie_search/EntropySearch/test.mat')
        #scipy.io.savemat(here+'/../../../entropie_search/EntropySearch/test.mat', dict(logP_=logP, dlogPdM_=dlogPdMu, dlogPdV_=dlogPdSigma, ddlogPdMdM_=dlogPdMudMu))
        bo_dummy = Dummy()
        bo_dummy.X_upper = self.X_upper
        bo_dummy.X_lower = self.X_lower
        bo_dummy.acquisition_fkt = entropy
        bo_dummy.dims = 1
        bo_dummy.model = self.model
        entropy.logP = dict_P["logP"]
        entropy.dlogPdMu = dict_P["dlogPdM"]
        entropy.dlogPdSigma = dict_P["dlogPdV"]
        entropy.dlogPdMudMu = dict_P["ddlogPdMdM"]
        
        entropy.W = dict_P["W"]
        dict_kernel = scipy.io.loadmat(here+'/../../../entropie_search/EntropySearch/kernel.mat')
        entropy.K = entropy.model.K
        entropy.cK = entropy.model.cK.T
        a =  dict_kernel["cK"]
        entropy.kbX = entropy.model.kernel.K(entropy.zb,entropy.model.X)
        entropy.logP = np.reshape(entropy.logP, (entropy.logP.shape[0], 1))
        print entropy(np.array([[0.5]]));
        dest_folder =here+"/tmp/test_entopy/vis"
        if dest_folder is not None:
            try:
                os.makedirs(dest_folder)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
        Visualization(bo_dummy, dest_folder=dest_folder, prefix=datetime.now().strftime("%Y.%m.%d.%H.%M.%S"), new_x = None, X=self.x, Y=self.y, acq_method = True, obj_method = False, model_method = True, )
if __name__=="__main__":
    unittest.main()
