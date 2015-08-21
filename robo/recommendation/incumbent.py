
import numpy as np


def compute_incumbent(model, X_lower=None, X_upper=None, inc=None):
    """
        Determines the incumbent as the best configuration with the lowest observation that has been found so far
    """
    best = np.argmin(model.Y)
    incumbent = model.X[best]
    incumbent_value = model.Y[best]

    return incumbent, incumbent_value



# #TODO: refactoring
# def _compute_incumbent(model, num_of_points):
# 
#         
#     # Create sobol grid over the input space
#     sobol_grid = i4_sobol_generate(self._comp.shape[1], num_of_points, 1)
# 
#     #optimization bounds
#     opt_bounds = zip(model.X_lower, model.X_upper)
# 
#     minima = np.zeros([sobol_grid.shape[1], sobol_grid.shape[0]])
#     minima_values = np.zeros([sobol_grid.shape[1]])
# 
#     # Optimize each point of the sobol grid locally
#     for i in xrange(0, sobol_grid.shape[1]):
#         point, func, info = spo.fmin_l_bfgs_b(obj, sobol_grid[:, i].flatten(), bounds=opt_bounds)
#         minima[i] = point
#         minima_values[i] = func
#     return minima, minima_values
# 
# #TODO: refactoring
# def optimize_gp_mcmc(self, x):
#     '''
#     Computes the sum of mean and standard deviation of each Gaussian process in x.
#     Args:
#         x: a numpy vector (not a matrix)
#         model_list: a list of Gaussian processes
#     Returns:
#         the value or if gradients is True additionally a numpy vector containing the gradients
#     '''
# 
#     x = np.array([x])
#     #will contain mean and standard deviation prediction for each GP
#     mean_std = np.zeros([self._mcmc_iters, 2, 1])
#     for i in range(0, self._mcmc_iters):
#         mean_std[i] = self._models[i].predict(x, True)
# 
#     #take square root to get standard deviation
#     mean_std[:, 1] = np.sqrt(mean_std[:, 1])
# 
#     mean_std_gradients = np.zeros([self._mcmc_iters, x.shape[1]])
# 
#     for i in range(0, self._mcmc_iters):
#         mg, vg = self._models[i].getGradients(x[0])
#         #getGradient returns the gradient of the variance - to get the gradients of the standard deviation
#         # we need to apply chain rule (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
#         stdg = 0.5 * vg / mean_std[i, 1]
#         mean_std_gradients[i] = (mg + stdg) / self._mcmc_iters
# 
#     #since we want to minimize, we have to turn the sign of the gradient
#     return (np.sum(mean_std) / self._mcmc_iters, np.sum(mean_std_gradients, axis=0))