'''
Created on Jun 25, 2015

@author: Aaron Klein
'''
import GPy
import time
import numpy as np
import matplotlib.pyplot as plt

from robo.benchmarks.branin import branin, get_branin_bounds
from robo.benchmarks.hartmann6 import hartmann6, get_hartmann6_bounds
from robo.benchmarks.goldstein_price import goldstein_price, get_goldstein_price_bounds
from robo.models.gpy_model_mcmc import GPyModelMCMC
from robo.models import gpy_model.GPyModel

from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def burnin(objective_function, kernel, n_dims, X_lower, X_upper, ax1, ax2, color, label, n_folds=3, num_points=20):

    grid = [100, 200, 300, 400, 500]
    perf_curve = np.zeros([len(grid), n_folds])
    time_curve = np.zeros([len(grid), n_folds])

    for n, burnin_size in enumerate(grid):

        X = np.random.rand(num_points, n_dims) * (X_upper - X_lower) + X_lower
        y = objective_function(X)

        model = GPyModelMCMC(kernel, noise_variance=None, burnin=burnin_size, chain_length=100, n_hypers=10)

        kf = cross_validation.KFold(num_points, n_folds=n_folds)

        for k, (train_index, test_index) in enumerate(kf):
            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            s = time.time()
            model.train(X_train, y_train)
            t = time.time() - s
            y_test_hat, _ = model.predict(X_test)

            perf_curve[n, k] = mean_squared_error(y_test, y_test_hat.mean(axis=0))
            time_curve[n, k] = t

    grid = np.array(grid)

    ax1.errorbar(np.array(grid), perf_curve.mean(axis=1), np.sqrt(perf_curve.var(axis=1)), fmt=color, label=label)
    ax1.set_xlabel("# burnin steps")
    ax1.set_ylabel('MSE')

    ax2.errorbar(grid, time_curve.mean(axis=1), np.sqrt(time_curve.var(axis=1)), fmt=color, label=label)
    ax2.set_xlabel("# burnin steps")
    ax2.set_ylabel('Time')

    return ax1, ax2


def hypers(objective_function, kernel, n_dims, X_lower, X_upper, ax1, ax2, color, label, n_folds=3, num_points=20):

    grid = [10, 20, 50, 80, 100]
    perf_curve = np.zeros([len(grid), n_folds])
    time_curve = np.zeros([len(grid), n_folds])

    for n, n_hypers in enumerate(grid):

        X = np.random.rand(num_points, n_dims) * (X_upper - X_lower) + X_lower
        y = objective_function(X)

        model = GPyModelMCMC(kernel, noise_variance=None, burnin=100, chain_length=800, n_hypers=n_hypers)

        kf = cross_validation.KFold(num_points, n_folds=n_folds)

        for k, (train_index, test_index) in enumerate(kf):
            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            s = time.time()
            model.train(X_train, y_train)
            t = time.time() - s
            y_test_hat, _ = model.predict(X_test)

            perf_curve[n, k] = mean_squared_error(y_test, y_test_hat.mean(axis=0))
            time_curve[n, k] = t

    grid = np.array(grid)

    ax1.errorbar(np.array(grid), perf_curve.mean(axis=1), np.sqrt(perf_curve.var(axis=1)), fmt=color, label=label)
    ax1.set_xlabel("# burnin steps")
    ax1.set_ylabel('MSE')

    ax2.errorbar(grid, time_curve.mean(axis=1), np.sqrt(time_curve.var(axis=1)), fmt=color, label=label)
    ax2.set_xlabel("# burnin steps")
    ax2.set_ylabel('Time')

    return ax1, ax2


def opt_vs_mcmc_data(objective_function, kernel, n_dims, X_lower, X_upper, ax1, ax2, ax3, n_folds=5):

    grid = [10, 20, 30, 40, 50]
    perf_curve_multi = np.zeros([len(grid), n_folds])
    perf_curve_single = np.zeros([len(grid), n_folds])

    r2_curve_multi = np.zeros([len(grid), n_folds])
    r2_curve_single = np.zeros([len(grid), n_folds])

    time_curve_multi = np.zeros([len(grid), n_folds])
    time_curve_single = np.zeros([len(grid), n_folds])

    for n, num_points in enumerate(grid):
        X = np.random.rand(num_points, n_dims) * (X_upper - X_lower) + X_lower
        y = objective_function(X)

        model = GPyModelMCMC(kernel, noise_variance=None, burnin=200, chain_length=100, n_hypers=10)

        single_model = gpy_model(kernel)

        kf = cross_validation.KFold(num_points, n_folds=n_folds)

        for k, (train_index, test_index) in enumerate(kf):
            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            s = time.time()
            single_model.train(X_train, y_train)
            t = time.time() - s
            time_curve_single[n, k] = t

            y_test_hat, _ = single_model.predict(X_test)
            perf_curve_single[n, k] = mean_squared_error(y_test, y_test_hat)
            r2_curve_single[n, k] = r2_score(y_test, y_test_hat)

            s = time.time()
            model.train(X_train, y_train)
            t = time.time() - s
            y_test_hat, _ = model.predict(X_test)

            perf_curve_multi[n, k] = mean_squared_error(y_test, y_test_hat.mean(axis=0))
            r2_curve_multi[n, k] = r2_score(y_test, y_test_hat.mean(axis=0))
            time_curve_multi[n, k] = t

    grid = np.array(grid)

    ax1.errorbar(np.array(grid), perf_curve_multi.mean(axis=1), np.sqrt(perf_curve_multi.var(axis=1)), fmt="blue", label="MCMC")
    ax1.errorbar(grid, perf_curve_single.mean(axis=1), np.sqrt(perf_curve_single.var(axis=1)), fmt="red", label="Opt")
    ax1.set_xlabel("# data points")
    ax1.set_ylabel('MSE')

    ax2.errorbar(np.array(grid), time_curve_multi.mean(axis=1), np.sqrt(time_curve_multi.var(axis=1)), fmt="blue", label="MCMC")
    ax2.errorbar(grid, time_curve_single.mean(axis=1), np.sqrt(time_curve_single.var(axis=1)), fmt="red", label="Opt")
    ax2.set_xlabel("# data points")
    ax2.set_ylabel('Time')

    ax3.errorbar(np.array(grid), r2_curve_multi.mean(axis=1), np.sqrt(r2_curve_multi.var(axis=1)), fmt="blue", label="MCMC")
    ax3.errorbar(grid, r2_curve_single.mean(axis=1), np.sqrt(r2_curve_single.var(axis=1)), fmt="red", label="Opt")
    ax3.set_xlabel("# data points")
    ax3.set_ylabel('r2 score')


def plot_log_likelihood(burnin=200, chain_length=100, n_hypers=10):
    num_points = 10
    X_lower, X_upper, n_dims = get_branin_bounds()
    X = np.random.rand(num_points, n_dims)
    y = branin(X)
    kernel = GPy.kern.Matern52(input_dim=n_dims)
    model = GPyModelMCMC(kernel, noise_variance=None, burnin=burnin, chain_length=chain_length, n_hypers=n_hypers)
    model.train(X, y)

    m = GPy.models.GPRegression(X, y, kernel)
#     hmc = GPy.inference.mcmc.hmc.HMC(m, stepsize=5e-2)
#     samples = hmc.sample(num_samples=100)
#     log_likelihood = np.zeros([model.mcmc_chain.shape[0]])
#     for j, sample in enumerate(model.mcmc_chain):
#         for i in range(len(sample) - 1):
#             kernel.parameters[i][0] = sample[i]
#         m.likelihood.variance[:] = sample[-1]
#         log_likelihood[j] = m.log_likelihood()

    xaxis = np.linspace(30, 100, 1000)
    yaxis = np.linspace(1, 20, 1000)

    grid_log_likelihood = np.zeros([1000, 1000])
    for i, x in enumerate(xaxis):
        for j, y in enumerate(yaxis):
            kernel.parameters[0][0] = xaxis[i]
            kernel.parameters[1][0] = yaxis[j]
            grid_log_likelihood[i, j] = m.log_likelihood()

    plt.plot(model.mcmc_chain[:burnin, 0], model.mcmc_chain[:burnin, 1], "r+")
    plt.plot(model.mcmc_chain[burnin:, 0], model.mcmc_chain[burnin:, 1], "ro")
    plt.plot(model.samples[:, 0], model.samples[:, 1], "bo")

    cs = plt.contour(xaxis, yaxis, grid_log_likelihood)
    plt.colorbar(cs)
    plt.xlabel(kernel.parameter_names()[0])
    plt.ylabel(kernel.parameter_names()[1])

    plt.savefig("log_likelihood.png")


# f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
# opt_vs_mcmc_data(ax1, ax2, ax3)
# plt.legend()
# plt.savefig("opt_vs_mcmc_data.png")

colors = ["blue", "green", "orange"]
labels = ["Matern52", "Matern32", "RBF"]

# Branin

X_lower, X_upper, n_dims = get_branin_bounds()
kernels = []
kernels.append(GPy.kern.Matern52(input_dim=n_dims))
kernels.append(GPy.kern.Matern32(input_dim=n_dims))
kernels.append(GPy.kern.RBF(input_dim=n_dims))

f, (ax1, ax2) = plt.subplots(2, sharex=True)
print "Burnin branin"
for i, kernel in enumerate(kernels):
    burnin(branin, kernel, n_dims, X_lower, X_upper, ax1, ax2, colors[i], labels[i])
plt.legend()
plt.savefig("branin_burnin_steps.png")
plt.clf()

print "Hypers branin"
f, (ax1, ax2) = plt.subplots(2, sharex=True)
for i, kernel in enumerate(kernels):
    hypers(branin, kernel, n_dims, X_lower, X_upper, ax1, ax2, colors[i], labels[i])
plt.legend()
plt.savefig("branin_n_hypers.png")

# Hartmann 6

X_lower, X_upper, n_dims = get_hartmann6_bounds()
kernels = []
kernels.append(GPy.kern.Matern52(input_dim=n_dims))
kernels.append(GPy.kern.Matern32(input_dim=n_dims))
kernels.append(GPy.kern.RBF(input_dim=n_dims))

print "Burnin hartmann"

f, (ax1, ax2) = plt.subplots(2, sharex=True)
for i, kernel in enumerate(kernels):
    burnin(hartmann6, kernel, n_dims, X_lower, X_upper, ax1, ax2, colors[i], labels[i])
plt.legend()
plt.savefig("hartmann6_burnin_steps.png")
plt.clf()

print "Hypers hartmann"

f, (ax1, ax2) = plt.subplots(2, sharex=True)
for i, kernel in enumerate(kernels):
    hypers(hartmann6, kernel, n_dims, X_lower, X_upper, ax1, ax2, colors[i], labels[i])
plt.legend()
plt.savefig("hartmann6_n_hypers.png")

# GoldsteinPrice

print "Burnin goldstein price"
X_lower, X_upper, n_dims = get_goldstein_price_bounds()
kernels = []
kernels.append(GPy.kern.Matern52(input_dim=n_dims))
kernels.append(GPy.kern.Matern32(input_dim=n_dims))
kernels.append(GPy.kern.RBF(input_dim=n_dims))

f, (ax1, ax2) = plt.subplots(2, sharex=True)
for i, kernel in enumerate(kernels):
    burnin(goldstein_price, kernel, n_dims, X_lower, X_upper, ax1, ax2, colors[i], labels[i])
plt.legend()
plt.savefig("goldstein_price_burnin_steps.png")
plt.clf()

print "hypers goldstein price"

f, (ax1, ax2) = plt.subplots(2, sharex=True)
for i, kernel in enumerate(kernels):
    hypers(goldstein_price, kernel, n_dims, X_lower, X_upper, ax1, ax2, colors[i], labels[i])
plt.legend()
plt.savefig("goldstein_price_n_hypers.png")
#plot_log_likelihood()

