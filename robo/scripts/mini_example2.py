import GPy
import matplotlib; matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt;
import numpy as np
import random

from robo.models import GPyModel
from robo.acquisition import EI
from robo.maximize import stochastic_local_search

def objective_function(x):
    return  np.sin(3*x) * 4*(x-1)* (x+2)

X_lower = np.array([0])
X_upper = np.array([6])
dims = 1
maximizer = stochastic_local_search

kernel = GPy.kern.Matern52(input_dim=dims)
model = GPyModel(kernel, optimize=True, noise_variance = 1e-4, num_restarts=10)
acquisition_func = EI(model, X_upper= X_upper, X_lower=X_lower, par =0.1) #par is the minimum improvement


X = np.empty((1, dims))
for i in xrange(dims):
    X[0,i] = random.random() * (X_upper[i] - X_lower[i]) + X_lower[i];
    x = np.array(X)

Y = objective_function(X)
for i in xrange(10):
    model.train(X, Y)
    acquisition_func.update(model)
    new_x = maximizer(acquisition_func, X_lower, X_upper)
    new_y = objective_function(np.array(new_x))
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)
    fig = plt.figure()
    ax1 =  fig.add_subplot(1, 1, 1)
    plotting_range = np.linspace(X_lower[0], X_upper[0], num=1000)
    ax1.plot(plotting_range, objective_function(plotting_range[:, np.newaxis]), color='b', linestyle="--")
    _min_y1, _max_y1 = ax1.get_ylim()
    model.visualize(ax1, X_lower[0], X_upper[0])
    _min_y2, _max_y2 = ax1.get_ylim()
    ax1.set_ylim(min(_min_y1, _min_y2), max(_max_y1, _max_y2))
    mu, var = model.predict(new_x)
    ax1.plot(new_x[0], mu[0], "r." , markeredgewidth=5.0)
    ax2 = acquisition_func.plot(fig, X_lower[0], X_upper[0], plot_attr={"color":"red"}, resolution=1000)

    plt.show(block=True)