'''
Created on Jun 9, 2015

@author: Aaron Klein
'''

import matplotlib.pyplot as plt
import numpy as np
import GPy

from robo.models import gpy_model.GPyModel
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.acquisition.EI import EI
from robo.recommendation.incumbent import compute_incumbent


def objective_function(x):
        return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)

X_lower = np.array([0])
X_upper = np.array([6])

dims = 1

#X = np.random.rand(14) * X_upper
X = np.array([0.1, 0.3, 2.0, 4.1])
X = X[:, np.newaxis]

y = objective_function(X)

kernel = GPy.kern.Matern52(input_dim=dims, lengthscale=0.01)
model = gpy_model(kernel, optimize=True, noise_variance=1e-8, num_restarts=10)
model.train(X, y)


es = EnvEntropySearch(None, None, n_representer=100, n_hals_vals=0, n_func_samples=0)

ei = EI(model, X_lower, X_upper, compute_incumbent)

test_data = np.array([np.random.randn()])

#callback = ei.__call__

representers = es._sample_representers(ei, n_representers=100, n_dim=1, burnin_steps=10, mcmc_steps=100)

representers = np.sort(representers, axis=0)

pmin = es._compute_pmin(model, representers, 1000)

grid = np.arange(X_lower, X_upper, 0.1)[:, np.newaxis]

mean, var = model.predict(grid, full_cov=False)

f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(grid, mean, "b")
axarr[0].fill_between(grid[:, 0], mean + 2 * np.sqrt(var), mean - 2 * np.sqrt(var), facecolor='red')


axarr[0].plot(X[:, 0], y,"k+")
axarr[0].plot(representers, np.ones(representers.shape) * (-40), "mo")
axarr[1].bar(representers, pmin, 0.05, color="green")

color = ["orange", "purple", "black", "yellow", "magenta"]
#for i in range(func_samples.shape[0]):
    #plt.plot(representers, func_samples[i, :], color[i % 5], marker="+")

plt.savefig("pmin_approximation.png")
