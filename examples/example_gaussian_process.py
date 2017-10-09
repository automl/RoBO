import matplotlib.pyplot as plt
import gpflow
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from robo.initial_design.init_random_uniform import init_random_uniform
from robo.models.gaussian_process import GaussianProcess

def f(x):
    return np.sinc(x * 10 - 5).sum(axis=1)

rng = np.random.RandomState(42)
X = init_random_uniform(np.zeros(1), np.ones(1), 20, rng)
y = f(X)

kernel = gpflow.kernels.Matern52(1, lengthscales=0.3)
prior = gpflow.priors.Gamma(2,3)

model = GaussianProcess(kernel=kernel, prior = prior, normalize_input=True, normalize_output=True)

model.train(X,y)
x = np.linspace(0, 1, 100)[:, None]
vals = f(x)

mean_pred, var_pred = model.predict(x)
std_pred = np.sqrt(var_pred)
plt.grid()
plt.plot(x[:, 0], vals, label="true", color="black")
plt.plot(X[:, 0], y, "ro")
plt.plot(x[:, 0], mean_pred, label="SGLD", color="green")
plt.fill_between(x[:, 0],mean_pred[:,0] + std_pred[:,0], mean_pred[:,0] - std_pred[:,0], alpha=0.2, color="green")
plt.show()