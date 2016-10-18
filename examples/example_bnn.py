import matplotlib.pyplot as plt
import numpy as np
import logging

from robo.models.bnn import BayesianNeuralNetwork
from robo.initial_design.init_random_uniform import init_random_uniform


def f(x):
    return np.sinc(x * 10 - 5).sum(axis=1)[:, None]

logging.basicConfig(level=logging.INFO)

rng = np.random.RandomState(42)

X = init_random_uniform(np.zeros(1), np.ones(1), 50, rng)
Y = f(X)

#model = BayesianNeuralNetwork(sampling_method="sgld", l_rate=1e-3, burn_in=30000, n_iters=100000)
#model.train(X, Y)

x = np.linspace(0, 1, 100)[:, None]

vals = f(x)

#mean_pred, var_pred = model.predict(x)

#std_pred = np.sqrt(var_pred)

plt.grid()
#plt.plot(x[:, 0], mean_pred[:, 0], label="SGLD", color="red")
plt.plot(x[:, 0], vals[:, 0], label="true", color="green")
plt.plot(X[:, 0], Y[:, 0], "ro")
#plt.fill_between(x[:, 0], mean_pred[:, 0] + std_pred[:, 0], mean_pred[:, 0] - std_pred[:, 0], alpha=0.2, color="orange")

model = BayesianNeuralNetwork(sampling_method="sghmc", l_rate=1e-4, burn_in=20000, n_iters=50000)
model.train(X, Y)

vals = f(x)

mean_pred, var_pred = model.predict(x)

std_pred = np.sqrt(var_pred)

plt.plot(x[:, 0], mean_pred[:, 0], label="SGHMC", color="blue")
plt.fill_between(x[:, 0], mean_pred[:, 0] + std_pred[:, 0], mean_pred[:, 0] - std_pred[:, 0], alpha=0.2, color="blue")

plt.legend()
plt.show()




