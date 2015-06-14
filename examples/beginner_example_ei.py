'''
Created on Apr 17, 2015

@author: Aaron Klein
'''
import GPy
import numpy as np

import matplotlib.pyplot as plt

from robo.models.GPyModel import GPyModel
from robo.acquisition.EI import EI
from robo.acquisition.Entropy import Entropy
from robo.maximizers.maximize import grid_search
from robo.recommendation.incumbent import compute_incumbent
from robo.visualization.plotting import plot_model, plot_objective_function, plot_acquisition_function
from robo import BayesianOptimization


def objective_function(x):
        return np.sin(x) + 0.1 * np.cos(10 * x)

X_lower = np.array([0])
X_upper = np.array([6])

dims = 1

kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
acquisition_func = EI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)
maximizer = grid_search

bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          X_lower=X_lower,
                          X_upper=X_upper,
                          dims=dims,
                          objective_fkt=objective_function)

bo.run(num_iterations=8)

X, Y = bo.get_observations()
X = X[:-1]
Y = Y[:-1]
model = bo.get_model()

print model.X

f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1 = plot_model(model, X_lower, X_upper, ax1)
ax1 = plot_objective_function(objective_function, X_lower, X_upper, X, Y, ax1)

ax2 = plot_acquisition_function(acquisition_func, X_lower, X_upper, ax2)

plt.legend()
plt.savefig("bo.png")


