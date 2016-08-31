
import numpy as np

from robo.initial_design.init_random_uniform import init_random_uniform


def extrapolative_initial_design(X_lower, X_upper, is_env, N):

    # Create grid for the system size
    idx = is_env == 1
    g = np.array([X_upper[idx] / float(i) for i in [4, 8, 16, 32]])[:, 0]

    X = init_random_uniform(X_lower, X_upper, N)

    X[:, is_env == 1] = \
        np.tile(g, np.ceil(X.shape[0] / 4.))[:X.shape[0], np.newaxis]

    return X
