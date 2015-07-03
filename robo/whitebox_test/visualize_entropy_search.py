'''
Created on Jul 3, 2015

@author: Aaron Klein
'''

import GPy
import numpy as np
import matplotlib.pyplot as plt

from robo.models.GPyModel import GPyModel
from robo.acquisition.EntropyMC import EntropyMC
from robo.recommendation.incumbent import compute_incumbent
from robo.visualization.plotting import plot_model, plot_projected_model,\
    plot_objective_function


def objective_function(x):
        return  np.array(np.sin(3 * x[:, 0]) * 4 * (x[:, 0] - 1) * (x[:, 0] + 2))[:, np.newaxis]


def cost_function(x):
        return  x[:, 1, np.newaxis]


X_lower = np.array([0])
X_upper = np.array([6])

dims = 1

X = np.random.rand(5, 1) * X_upper

y = objective_function(X)

kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True)
model.train(X, y)

es = EntropyMC(model, X_lower, X_upper, compute_incumbent)
es.update(model)

# Plot model
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1 = plot_model(model, X_lower, X_upper, ax1)
ax1 = plot_objective_function(objective_function, X_lower, X_upper, X, y, ax1)

rep = es.zb

xaxis = np.arange(X_lower[0], X_upper[0], 0.1)

grid_log_ei = np.zeros([xaxis.shape[0]])
for i, x in enumerate(xaxis):
        grid_log_ei[i] = np.e ** es.sampling_acquisition(np.array([xaxis[i]]))

ax2.plot(rep, np.zeros(rep.shape), "bo")
cs = ax2.plot(xaxis, grid_log_ei, "g")


ax3.plot(rep[:, 0], np.zeros(rep.shape[0]), "bo")
ax3.bar(rep[:, 0], es.pmin, 0.05, color="orange")


# Plot halluzinated values, representers, some function samples and fantasised pmin

idx = np.argsort(rep, axis=0)

ax4.plot(rep[idx, 0], es.f[idx, 0], "b")
ax4.plot(rep[idx, 0], es.f[idx, 100], "g")
ax4.plot(rep[idx, 0], es.f[idx, 200], "y")
ax4.plot(rep[idx, 0], es.f[idx, 300], "r")
ax4.plot(rep[idx, 0], es.f[idx, 400], "k")
ax4.plot(rep[idx, 0], es.f[idx, 800], "m")
print rep[idx, 0]
plt.savefig("robo/whitebox_test/plots_entropy_search/es.png")
plt.clf()

#TODO:Plot acquisition function
