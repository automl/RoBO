import logging
import numpy as np
import pyGPs
from robo.models.base_model import BaseModel
import scipy.linalg as spla

logger = logging.getLogger(__name__)

class PyGPModel(BaseModel):
    def __init__(self, kernel, num_restarts=10, *args, **kwargs):
        self.kernel = kernel
        self.num_restarts = num_restarts
        self.m = None

    def train(self, X, Y, optimize=True):
        self.X = X
        self.Y = Y
        if X.size == 0 or Y.size == 0:
            return
        self.m = pyGPs.GPR()
        self.m.setPrior(kernel=self.kernel);

        if optimize:
            self.m.setOptimizer("Minimize", num_restarts=self.num_restarts)
            self.m.optimize(X, Y)
            logger.debug("Covariance Function parameters")
            logger.debug(self.m.covfunc.hyp)
            logger.debug(self.m.likfunc.hyp)
        else:
            self.m.setData(X, Y)

    def predict_variance(self, X1, X2):
        if self.m == None:
            logger.error("ERROR: Model has to be trained first.")
            return None
        LX1 = spla.cho_solve((self.m.posterior.L, True), self.kernel.getCovMatrix(self.X, X1, "cross"))
        LX2 = spla.cho_solve((self.m.posterior.L, True), self.kernel.getCovMatrix(self.X, X2, "cross"))
        var = self.kernel.getCovMatrix(X1, X2, "cross") - np.dot(LX1.T, LX2)
        return var

    def predict(self, X, full_cov=False):
        if self.m == None:
            logger.error("ERROR: Model has to be trained first.")
            return None

        mean, var, _, _, _ = self.m.predict(X)
    	if full_cov:
            covar = self.kernel.getCovMatrix(self.X, X, "cross")
            Lkstar = spla.cho_solve((self.m.posterior.L, True), covar)
    	    var = self.kernel.getCovMatrix(X, X, "cross") - np.dot(Lkstar.T, Lkstar)
        return mean, var

    def predictive_gradients(self, Xnew, X=None):
        raise NotImplementedError()

    def sample(self, X, size=10):
        """
        samples from the GP at values X size times.
        """
        Omega = np.random.standard_normal([X.shape[1], size])
        return np.dot(self.m.posterior.L, Omega)

    def negative_log_likelihood(self, kernel):
	model = pyGPs.GPR()  
	model.setPrior(kernel=kernel)    
	model.getPosterior(self.X, self.Y)
	return model.nlZ

    def visualize(self, ax, plot_min, plot_max):
        self.m.plot()
