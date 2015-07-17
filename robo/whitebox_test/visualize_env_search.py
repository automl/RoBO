'''
Created on Jul 1, 2015

@author: Aaron Klein
'''

import GPy
import numpy as np
import matplotlib.pyplot as plt

from robo.models.GPyModel import GPyModel
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.recommendation.incumbent import compute_incumbent
from robo.visualization.plotting import plot_model, plot_projected_model,\
    plot_model_2d, plot_acquisition_function_2d


def objective_function(x):
        return  np.array(np.sin(3 * x[:, 0]) * 4 * (x[:, 0] - 1) * (x[:, 0] + 2))[:, np.newaxis]


def cost_function(x):
        return  x[:, 1, np.newaxis]


X_lower = np.array([0, 6])
X_upper = np.array([6, 10])

dims = 2

X = np.random.rand(20, 2) * X_upper

y = objective_function(X)

c = cost_function(X)

kernel = GPy.kern.Matern52(input_dim=dims)
cost_kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True)
cost_model = GPyModel(cost_kernel, optimize=True)
model.train(X, y)
cost_model.train(X, c)

is_env = np.array([0, 1])

env_es = EnvEntropySearch(model, cost_model, X_lower, X_upper, compute_incumbent, is_env)
env_es.update(model, cost_model)
rep = env_es.zb

# Plot model and cost model
#TODO: plot 2D
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=False)
#ax1 = plot_model_2d(model, X_lower, X_upper, ax1, 0.1)
#ax1.plot(X[:, 0] * 10, X[:, 1]* 10, 'ko')

ax1 = plot_projected_model(model, X_lower, X_upper, ax1, X_upper[0], 0.1)
ax3 = plot_model_2d(cost_model, X_lower, X_upper, ax3, 0.1)
ax4 = plot_acquisition_function_2d(env_es, X_lower, X_upper, ax4, 0.1)
ax3.plot(rep[:, 0]*10, rep[:, 1]*10, "ko")
ax2.plot(rep[:, 0], np.zeros(rep.shape[0]), "bo")
ax2.bar(rep[:, 0], env_es.pmin, 0.05, color="orange")

plt.savefig("/home/aaron/devel/git/RoBO/robo/whitebox_test/plots_env_search/models.png")
plt.clf()


xaxis = np.arange(X_lower[0], X_upper[0], 0.1)
yaxis = np.arange(X_lower[1], X_upper[1], 0.1)
grid_log_ei = np.zeros([xaxis.shape[0], yaxis.shape[0]])
for i, x in enumerate(xaxis):
    for j, y in enumerate(yaxis):
        grid_log_ei[i, j] = env_es.sampling_acquisition(np.array([xaxis[i], yaxis[j]]))

plt.plot(rep[:, 0], rep[:, 1], "bo")
cs = plt.contour(xaxis, yaxis, grid_log_ei)
plt.savefig("/home/aaron/devel/git/RoBO/robo/whitebox_test/plots_env_search/representers.png")
plt.clf()

#TODO:Plot current pmin

#TODO:Plot innovated model

#TODO:Plot halluzinated values, representers, some function samples and fantasised pmin

#TODO:Plot acquisition function
