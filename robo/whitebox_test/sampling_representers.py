'''
Created on Jun 9, 2015

@author: Aaron Klein
'''

import matplotlib.pyplot as plt
import numpy as np
import GPy

from robo.models import gpy_model.GPyModel
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.acquisition.LogEI import LogEI
from robo.recommendation.incumbent import compute_incumbent


def objective_function(x):
        return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)

X_lower = np.array([0])
X_upper = np.array([6])

dims = 1

X = np.random.rand(10) * X_upper
X = X[:, np.newaxis]

y = objective_function(X)

kernel = GPy.kern.Matern52(input_dim=dims)
model = gpy_model(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
model.train(X, y)

es = EnvEntropySearch(None, None, n_representer=100, n_hals_vals=0, n_func_samples=0)

ei = LogEI(model, X_lower, X_upper, compute_incumbent)

test_data = np.array([np.random.randn()])


representers = es._sample_representers(ei, n_representers=100, n_dim=1, burnin_steps=100, mcmc_steps=100)

grid = np.arange(X_lower, X_upper, 0.1)[:, np.newaxis]
grid_values = np.zeros([grid.shape[0]])
for i in range(grid.shape[0]):
    grid_values[i] = ei(grid[i])

plt.plot(grid[:, 0], grid_values, "g")

plt.plot(representers, np.zeros(representers.shape[0]), "b+")

plt.show()
plt.savefig("representers.png")
