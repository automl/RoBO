import numpy as np
import matplotlib.pyplot as plt

from robo.initial_design.init_random_uniform import init_random_uniform
from robo.models.bayesian_linear_regression import BayesianLinearRegression


def f(x):
    return 10 * x - 5 + np.random.randn() * 0.001

rng = np.random.RandomState(42)

X = init_random_uniform(np.zeros(1), np.ones(1), 20, rng)
y = f(X)[:, 0]

model = BayesianLinearRegression()

model.train(X, y, do_optimize=True)


X_test = np.linspace(0, 1, 100)[:, None]

fvals = f(X_test)[:, 0]

m, v = model.predict(X_test)


plt.plot(X, y, "ro")
plt.grid()
plt.plot(X_test[:, 0], fvals, "k--")
plt.plot(X_test[:, 0], m, "blue")
plt.fill_between(X_test[:, 0], m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.4)
plt.xlim(0, 1)
plt.show()