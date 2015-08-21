'''
Created on 18.07.2015

@author: Aaron Klein
'''

import GPy

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from robo.models import gpy_model.GPyModel
from robo.recommendation.incumbent import compute_incumbent
from robo.acquisition.EntropyMC import EntropyMC
from robo.acquisition.Entropy import Entropy
from robo.acquisition.EI import EI
from robo.acquisition.PI import PI


from IPython import embed
from robo.task.base_task import BaseTask
from robo.visualization.plotting import plot_model, plot_acquisition_function


class ExampleTask(BaseTask):
    def __init__(self):
        self.X_lower = np.array([-5])
        self.X_upper = np.array([5])
        self.n_dims = 1

    def objective_function(self, x):
        return np.array([[1.0], [0.05], [0.15]])


def compute_real_pmin(model, ax, Nb=50, Nf=2000, Np=1000):
    zb = np.linspace(-5, 5, Nb)[:, np.newaxis]
    W = norm.ppf(np.linspace(1. / (Np + 1), 1 - 1. / (Np + 1), Np))[np.newaxis, :]
    F = np.random.multivariate_normal(mean=np.zeros(Nb), cov=np.eye(Nb), size=Nf)
    Mb, Vb = model.predict(zb, full_cov=True)
    cVb = np.linalg.cholesky(Vb)
    f = np.add(np.dot(cVb, F.T).T, Mb).T
    ax.plot(zb[:, 0], f[:, 0], "b+")
    ax.plot(zb[:, 0], f[:, 1], "r+")
    ax.plot(zb[:, 0], f[:, 2], "g+")
    ax.plot(zb[:, 0], f[:, 3], "k+")

    mins = np.argmin(f, axis=0)

    c = np.bincount(mins)
    min_count = np.zeros((Nb,))
    min_count[:len(c)] += c
    pmin = (min_count / f.shape[1])[:, np.newaxis]

    pmin[np.where(pmin < 1e-70)] = 1e-70
    return zb, pmin, ax


def main():

    task = ExampleTask()

    kernel = GPy.kern.Matern52(input_dim=task.n_dims)
    model = gpy_model(kernel, optimize=False, noise_variance=1e-8, num_restarts=10)

    n_representer = 50
    n_hals_vals = 1000
    n_func_samples = 2000

    esmc = EntropyMC(model, task.X_lower, task.X_upper, compute_incumbent, Nb=n_representer, Nf=n_func_samples, Np=n_hals_vals)
    es = Entropy(model, task.X_lower, task.X_upper, Nb=n_representer, Np=n_hals_vals)
    ei = EI(model, task.X_lower, task.X_upper, compute_incumbent)
    pi = PI(model, task.X_lower, task.X_upper, compute_incumbent)

    X = np.array([[-1], [1], [2]])
    Y = task.objective_function(X)

    model.train(X, Y)
    es.update(model)
    ei.update(model)
    pi.update(model)
    esmc.update(model)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    plt.xlim((-5, 5))
    ax1 = plot_model(model, task.X_lower, task.X_upper, ax1)
    ax1.plot(X[:, 0], Y[:, 0], "ko")

    ax2.plot(esmc.zb[:, 0], np.zeros(esmc.zb.shape), "bo")
    ax2.bar(esmc.zb[:, 0], esmc.pmin, 0.05, color="orange")
    ax4.bar(es.zb[:, 0], es.logP, 0.05, color="purple")

    zb, pmin, ax1 = compute_real_pmin(model, ax1)
    ax2.plot(zb[:, 0], pmin, "r")

    ax3 = plot_acquisition_function(ei, task.X_lower, task.X_upper, ax3, color="green", label="EI")
    ax3 = plot_acquisition_function(pi, task.X_lower, task.X_upper, ax3, color="orange", label="PI")
    ax3 = plot_acquisition_function(es, task.X_lower, task.X_upper, ax3, color="purple", label="ES")
    ax3 = plot_acquisition_function(esmc, task.X_lower, task.X_upper, ax3, color="blue", label="ESMC")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
