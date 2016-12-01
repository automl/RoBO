import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

import robo.models.neural_network as robo_net
import robo.models.bagged_networks as bn
from robo.initial_design.init_random_uniform import init_random_uniform

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def f(x):
    return np.sinc(x * 10 - 5).sum(axis=1)[:, None]

rng = np.random.RandomState(42)

X = init_random_uniform(np.zeros(1), np.ones(1), 20, rng).astype(np.float32)
Y = f(X)

x = np.linspace(0, 1, 512, dtype=np.float32)[:, None]
vals = f(x).astype(np.float32)

plt.grid()
plt.plot(x[:, 0], f(x)[:, 0], label="true", color="green")
plt.plot(X[:, 0], Y[:, 0], "ro")


model = bn.BaggedNets(robo_net.SGDNet, num_models=16, bootstrap_with_replacement=True,
                      n_epochs=16384, error_threshold=1e-3,
                      n_units=[32, 32, 32], dropout=0,
                      batch_size=10, learning_rate=1e-3,
                      shuffle_batches=True)

m = model.train(X, Y)

mean_pred, var_pred = model.predict(x)
std_pred = np.sqrt(var_pred)

plt.plot(x[:, 0], mean_pred[:, 0], label="bagged nets", color="blue")
plt.fill_between(x[:, 0], mean_pred[:, 0] + std_pred[:, 0], mean_pred[:, 0] - std_pred[:, 0], alpha=0.2, color="blue")

plt.legend()
plt.show()
