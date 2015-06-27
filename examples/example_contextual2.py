"""
Contextual Bayesian optimization, example
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import GPy
import scipy
from robo.contextual_bayesian_optimization import ContextualBayesianOptimization
from robo.models.GPyModel import GPyModel
from robo.acquisition.UCB import UCB
from robo.maximizers.maximize import stochastic_local_search

_branin_k1 = 5.1/(4*np.pi*np.pi)
_branin_k2 = 5 / np.pi
_branin_k3 = 10 * (1-1/(8*np.pi))

def branin(u, v):
    """
    The branin function.
    :param u: limits: [-5, 10]
    :param v: limits: [0, 15]
    :return:
    """
    w = (v - _branin_k1 * u * u + _branin_k2 * u - 6)
    return w * w + _branin_k3 * np.cos(u) + 10


def branin_grad_u(u, v):
    """
    Gradient of the branin function.
    :param u: limits: [-5, 10]
    :param v: limits: [0, 15]
    :return:
    """
    return 2*(_branin_k2 - 2*_branin_k1*u)*(-_branin_k1*u*u+_branin_k2*u+v-6)-_branin_k2*np.sin(u)


def branin_grad_v(u, v):
    """
    Gradient of the branin function.
    :param u: limits: [-5, 10]
    :param v: limits: [0, 15]
    :return:
    """
    return 2*(-_branin_k1*u*u + _branin_k2*u + v - 6)


def branin_min_u(u):
    """
    Returns the position of the minimum for a fixed u
    :param u:
    :return: v
    """
    return _branin_k1 * u * u - _branin_k2 * u + 6

def branin_min_v(v):
    """
    Returns the position of the minimum for a fixed v
    :param v:
    :return: u
    """
    res = np.zeros(v.shape)
    for i, v_ in enumerate(v):
        def minfn(x):
            return branin(x, v_), branin_grad_u(x, v_)
        res[i, :] = scipy.optimize.minimize(fun=minfn, x0=np.array((0.5,)), jac=True, bounds=((-5,10),)).x[0]
    return res

def objective1(Z, S):
    """
    This is a 4 dimensional function based on two branin evaluations
    :param Z: context [0, 1]^2
    :param S: action [0, 1]^2
    :return: value for the point(s)
    """
    x2, x3 = S[:, 0:1], S[:, 1:2]
    x1, x4 = Z[:, 0:1], Z[:, 1:2]
    return branin(15*x1 - 5, 15*x2) * branin(15*x3 - 5, 15*x4)

def objective1_min_action(Z):
    """
    Calculates the location (action) of the minimum for a given context
    :param Z: context
    :return: location of minimum as tuple
    """
    x1, x4 = Z[:, 0:1], Z[:, 1:2]
    x2 = branin_min_u(15*x1 - 5) / 15
    x3 = (branin_min_v(15*x4) + 5) / 15
    return np.concatenate((x2, x3), axis=1)

def objective1_min(Z):
    """
    Calculates the minimum for a given context
    :param Z: context
    :return: value of the minimum
    """
    x1, x4 = Z[:, 0:1], Z[:, 1:2]
    u1 = 15*x1 - 5
    v2 = 15*x4
    return branin(u1, branin_min_u(u1)) * branin(branin_min_v(v2), v2)

# Create figure
fig = plt.figure(figsize=(15, 10))
plt.hold(True)
ax1 = plt.gca()

############################################
# Objective 1: Product of Branin functions #
############################################

# Defining the bounds and dimensions of the input space
S_lower = np.array([0, 0])
S_upper = np.array([1, 1])

X_lower = np.array([-np.inf, -np.inf, 0, 0])
X_upper = np.array([np.inf, np.inf, 1, 1])

dims_Z = 2
dims_S = 2

# Set the method that we will use to optimize the acquisition function
maximizer = stochastic_local_search

# Defining the method to model the objective function
kernel = GPy.kern.Matern52(input_dim=dims_Z + dims_S)
model = GPyModel(kernel, optimize=True, noise_variance=1e-2, num_restarts=10)

# The acquisition function that we optimize in order to pick a new x
acquisition_func = UCB(model, X_lower=X_lower, X_upper=X_upper, par=1.0)

# Context function acquires random values
def context_fkt():
    return np.random.uniform(size=(1,2))

bo = ContextualBayesianOptimization(acquisition_fkt=acquisition_func,
                                    model=model,
                                    maximize_fkt=maximizer,
                                    S_lower=S_lower,
                                    S_upper=S_upper,
                                    dims_Z=dims_Z,
                                    dims_S=dims_S,
                                    objective_fkt=objective1,
                                    context_fkt=context_fkt)

print "Result:", bo.run(num_iterations=25)

# Calculate regret
real_data = objective1_min(Z=bo.X[:, :bo.dims_S]).flatten()
pred_data = bo.Y.flatten()
regret = pred_data - real_data
cum_regret = np.cumsum(regret)
contextual_regret = cum_regret / np.arange(1, len(cum_regret) + 1)

ax1.set_title('Regret of product of Branin functions')
ax1.set_title('regret')
ax1.set_xlabel('time')
ax1.set_ylabel('regret')
plt1, = ax1.plot(regret, label='Regret')
plt2, = ax1.plot(contextual_regret, label='Contextual Regret')
ax1.legend(handles=(plt1, plt2))

plt.show(block=True)