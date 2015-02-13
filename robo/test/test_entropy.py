
import sys
import os
#sys.path.insert(0, '../')
import unittest
import errno
import numpy as np
import random
random.seed(12)
import GPy
import scipy
from robo.models import GPyModel
from robo.acquisition import EI, Entropy
from robo.visualization import Visualization
import matplotlib.pyplot as plt
from datetime import datetime
class Dummy(object):
    pass

here = os.path.abspath(os.path.dirname(__file__))

#@unittest.skip("skip first test\n")
class EITestCase1(unittest.TestCase):
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
        #ei_par_1 = Entropy(self.model, X_upper=X_upper, X_lower=X_lower,  derivative=True)
        #ei_par_2 = Entropy(self.model, X_upper=X_upper, X_lower=X_lower,  derivative=True)
        x_values = [0.62971589] 
        entropy.update(self.model)
        scipy.io.savemat(here+'/../../../entropie_search/EntropySearch/test.mat', dict(zb = entropy.zb,
                                                    lmb = entropy.lmb, 
                                                    logP = entropy.logP,
                                                    dlogPdM = entropy.dlogPdMu,
                                                    dlogPdV = entropy.dlogPdSigma,
                                                    ddlogPdMdM = entropy.dlogPdMudMu))
        out0 = np.array([ entropy(np.array([[x]]), derivative=True) for x in x_values])
        value0 = out0[:,0]
        
        
        print value0
        """
        out1 = np.array([ ei_par_1(np.array([[x]]), derivative=True) for x in x_values])
        value1 = out1[:,0]
        derivative1 = out1[:,1]
        
        out2 = np.array([ ei_par_2(np.array([[x]]), derivative=True) for x in x_values])
        value2 = out2[:,0]
        derivative2 = out2[:,1]
        assert(value0[0] <= 1e-5)
        assert(np.all(value0 >= value1))
        assert(np.all(value1 >= value2))
        assert(np.all(np.abs(derivative0) >= np.abs(derivative1)))
        assert(np.all(np.abs(derivative1) >= np.abs(derivative2)))
        ei_par_0.update(self.noise_model)
        out0_noise = np.array([ ei_par_0(np.array([[x]]), derivative=True) for x in x_values])
        value0_noise = out0[:,0]
        derivative0_noise = out0[:,1]
        assert(np.all(np.abs(value0_noise) >= np.abs(value0)))
        """
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


if __name__=="__main__":
    unittest.main()
