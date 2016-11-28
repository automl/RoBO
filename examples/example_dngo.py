import lasagne
import numpy as np
import matplotlib.pyplot as plt

from robo.initial_design.init_random_uniform import init_random_uniform
from robo.models.dngo import DNGO
from robo.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_unnormalization


def f(x):
    return np.sinc(x * 10 - 5).sum(axis=1)[:, None]

rng = np.random.RandomState(42)

X = init_random_uniform(np.zeros(1), np.ones(1), 20, rng)
y = f(X)[:, 0]


model = DNGO()

model.train(X, y)

predictions = lasagne.layers.get_output(model.network,
                                        zero_mean_unit_var_normalization(X, model.X_mean, model.X_std)[0],
                                        deterministic=True).eval()

predictions = zero_mean_unit_var_unnormalization(predictions, model.y_mean, model.y_std)

X_test = np.linspace(0, 1, 100)[:, None]
X_test_norm = zero_mean_unit_var_normalization(X_test, model.X_mean, model.X_std)[0]

# Get features from the net
layers = lasagne.layers.get_all_layers(model.network)
basis_funcs = lasagne.layers.get_output(layers[:-1], X_test_norm)[-1].eval()

fvals = f(X_test)[:, 0]

m, v = model.predict(X_test)

colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, min(50, model.n_units_3))])

for f in range(min(50, model.n_units_3)):
    plt.plot(X_test[:, 0], basis_funcs[:, f])
plt.grid()
plt.xlabel(r"Input $x$")
plt.ylabel(r"Basisfunction $\theta(x)$")
plt.show()


plt.plot(X, y, "ro")
plt.plot(X, predictions, "g+")
plt.grid()
plt.plot(X_test[:, 0], fvals, "k--")
plt.plot(X_test[:, 0], m, "blue")
plt.fill_between(X_test[:, 0], m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.4)
plt.xlim(0, 1)
plt.show()
