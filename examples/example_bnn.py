import matplotlib.pyplot as plt
import numpy as np
import logging

from robo.models.bnn import BayesianNeuralNetwork
from robo.initial_design.init_random_uniform import init_random_uniform


def f(x):
    return np.sinc(x * 10 - 5).sum(axis=1)

logging.basicConfig(level=logging.INFO)

rng = np.random.RandomState(42)

X = init_random_uniform(np.zeros(1), np.ones(1), 20, rng)
y = f(X)

model = BayesianNeuralNetwork(sampling_method="sgld",
                              l_rate=1e-4,
                              mdecay=0.05,
                              burn_in=3000,
                              n_iters=50000,
                              precondition=True,
                              normalize_input=True,
                              normalize_output=True)
model.train(X, y)

x = np.linspace(0, 1, 100)[:, None]

vals = f(x)

mean_pred, var_pred = model.predict(x)

std_pred = np.sqrt(var_pred)

plt.grid()

plt.plot(x[:, 0], vals, label="true", color="black")
plt.plot(X[:, 0], y, "ro")

plt.plot(x[:, 0], mean_pred, label="SGLD", color="green")
plt.fill_between(x[:, 0], mean_pred + std_pred, mean_pred - std_pred, alpha=0.2, color="green")

model = BayesianNeuralNetwork(sampling_method="sghmc",
                              l_rate=np.sqrt(1e-4),
                              mdecay=0.05,
                              burn_in=3000,
                              n_iters=50000,
                              precondition=True,
                              normalize_input=True,
                              normalize_output=True)
model.train(X, y)

mean_pred, var_pred = model.predict(x)

std_pred = np.sqrt(var_pred)

plt.plot(x[:, 0], mean_pred, label="SGHMC", color="blue")
plt.fill_between(x[:, 0], mean_pred + std_pred, mean_pred - std_pred, alpha=0.2, color="blue")

plt.legend()
plt.show()




