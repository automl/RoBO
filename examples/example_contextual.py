"""
Contextual Bayesian optimization, example
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import GPy
from robo import BayesianOptimizationError
from robo.contextual_bayesian_optimization import ContextualBayesianOptimization
from robo.models.GPyModel import GPyModel
from robo.acquisition.UCB import UCB
from robo.maximizers.maximize import stochastic_local_search
# from robo.util.loss_functions import logLoss
# from robo.util.visualization import Visualization
from robo.recommendation.incumbent import compute_incumbent
from scipy.interpolate import griddata

_branin_k1 = 5.1/(4*np.pi*np.pi)  # ~0.129185
_branin_k2 = 5 / np.pi  # ~1.591549
_branin_k3 = 10 * (1-1/(8*np.pi))  # ~9.602113
def branin(u, v):
    """
    The branin function.
    :param u: limits: [-5, 10]
    :param v: limits: [0, 15]
    :return:
    """
    w = (v - _branin_k1 * u * u + _branin_k2 * u - 6)
    return w * w + _branin_k3 * np.cos(u) + 10

def branin_min_u(u):
    """
    Returns the position of the minimum for a fixed u
    :param u:
    :return: v
    """
    return _branin_k1 * u * u - _branin_k2 * u + 6

def objective0(Z, S):
    """
    This resembles the 2 dimensional branin function, normalized to [0,1] with the first variable as context and the
    second as action variable.
    :param Z: context [0, 1]
    :param S: action [0, 1]
    :return: value for the point(s)
    """
    x1 = Z[:, 0:1]
    x2 = S[:, 0:1]
    return branin(15*x1 - 5, 15*x2)

def objective0_min(Z):
    """
    Returns the exact minimum for the objective function, given the context.
    :param Z: context [0, 1]
    :return: the value for the minimum
    """
    x1 = Z[:, 0:1]
    u = 15*x1 - 5
    v = branin_min_u(u)
    return branin(u, v)

def objective0_min_action(Z):
    """
    Returns the exact minimum for the objective function, given the context.
    :param Z: context [0, 1]
    :return: the value for the minimum
    """
    x1 = Z[:, 0:1]
    u = 15*x1 - 5
    v = branin_min_u(u)
    return v / 15

def objective0_plot(Z, S):
    """
    This resembles the 2 dimensional branin function, normalized to [0,1] with the first variable as context and the
    second as action variable.
    Visualization comfort version.
    :param Z: context [0, 1]
    :param S: action [0, 1]
    :return: value for the point(s)
    """
    return branin(15*Z - 5,  15*S)


# Defining the bounds and dimensions of the input space
S_lower = np.array([0])
S_upper = np.array([1])

X_lower = np.array([-np.inf, 0])
X_upper = np.array([np.inf, 1])

dims_Z = 1
dims_S = 1
dims_X = dims_Z + dims_S

# Set the method that we will use to optimize the acquisition function
maximizer = stochastic_local_search

# Construct kernel by a separate kernel for context and action each
kernel = GPy.kern.Matern52(input_dim=dims_Z, active_dims=list(range(dims_Z))) \
    * GPy.kern.Matern52(input_dim=dims_S, active_dims=list(range(dims_Z, dims_X)))
# Defining the method to model the objective function
model = GPyModel(kernel, optimize=True, noise_variance=1e-2, num_restarts=10)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = UCB(model, X_lower=X_lower, X_upper=X_upper, par=1.0)

# Context function acquires random values
def context_fkt():
    return np.random.uniform(size=(1,1))

bo = ContextualBayesianOptimization(acquisition_fkt=acquisition_func,
                                    model=model,
                                    maximize_fkt=maximizer,
                                    S_lower=S_lower,
                                    S_upper=S_upper,
                                    dims_Z=dims_Z,
                                    dims_S=dims_S,
                                    objective_fkt=objective0,
                                    context_fkt=context_fkt)

print "Result:", bo.run(num_iterations=32)

print "Finding optima"
xactions = np.linspace(0, 1, num=32)
yactions = np.array([bo.predict(np.array(xeval, ndmin=2)).flatten()[1] for xeval in xactions])
yactions_exact = np.array([objective0_min_action(Z=np.reshape(xeval, (1, 1))).flatten() for xeval in xactions])

# Create grid for contours
x = np.linspace(0, 1, num=200)
y = np.linspace(0, 1, num=200)
X, Y = np.meshgrid(x, y)
Z = objective0_plot(X, Y)

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))
plt.hold(True)
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

# Create real contour plot
CS = ax1.contour(X, Y, Z, levels=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4],
                 norm=matplotlib.colors.LogNorm())
ax1.clabel(CS, inline=1, fontsize=10)
ax1.set_title('function (payoff)')
ax1.set_xlabel('context')
ax1.set_ylabel('action')
ax1.set_ylim((0, 1))
imx, imy = np.mgrid[0:1:100j, 0:1:100j]
resampled = griddata((X.flatten(), Y.flatten()), Z.flatten(), (imx, imy))
implt = ax1.imshow(resampled.T, extent=(0, 1, 0, 1), interpolation='bicubic', origin='lower', cmap=plt.get_cmap('hot'),
                   norm=matplotlib.colors.LogNorm(), aspect='auto')
resampledLimits = (resampled.min(), resampled.max())
plt.colorbar(implt, ax=ax1)
# Add optimum
ax1.plot(xactions, yactions_exact)

# Create predicted contour plot
x = np.linspace(0, 1, num=50)
y = np.linspace(0, 1, num=50)
X, Y = np.meshgrid(x, y)
# Predict mean
mean, var = model.predict(np.transpose(np.array((X.flatten(order='C'), Y.flatten(order='C')))))
Z = np.reshape(mean, X.shape, order='C')

CS = ax2.contour(X, Y, np.maximum(Z, 1e-3), levels=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2,
                                                    1e3, 3e3, 1e4], norm=matplotlib.colors.LogNorm())
ax2.clabel(CS, inline=1, fontsize=10)
ax2.set_title('predicted function (payoff)')
ax2.set_xlabel('context')
ax2.set_ylabel('action')
ax2.set_ylim((0, 1))
imx, imy = np.mgrid[0:1:100j, 0:1:100j]
resampled = griddata((X.flatten(), Y.flatten()), Z.flatten(), (imx, imy))
implt = ax2.imshow(resampled.T, extent=(0, 1, 0, 1), interpolation='bicubic', origin='lower', cmap=plt.get_cmap('hot'),
                   norm=matplotlib.colors.LogNorm(), aspect='auto', vmin=resampledLimits[0], vmax=resampledLimits[1])
plt.colorbar(implt, ax=ax2)
ax2.plot(bo.X[:, 0], bo.X[:, 1], 'x')
# Add optimum
ax2.plot(xactions, yactions)

# Calculate regret
real_data = objective0_min(Z=bo.X[:, :bo.dims_S]).flatten()
pred_data = bo.Y.flatten()
regret = pred_data - real_data
cum_regret = np.cumsum(regret)
contextual_regret = cum_regret / np.arange(1, len(cum_regret) + 1)

ax3.set_title('regret')
ax3.set_xlabel('iteration')
ax3.set_ylabel('regret')
plt1, = ax3.plot(regret, label='Regret')
plt2, = ax3.plot(contextual_regret, label='Contextual Regret')
ax3.legend(handles=(plt1, plt2))

plt.tight_layout()
plt.savefig('contextual_branin.svg')

#plt.show(block=True)