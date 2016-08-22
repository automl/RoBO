'''
Created on Dec 21, 2015

@author: Aaron Klein
'''

import numpy as np

from robo.initial_design.init_random_uniform import init_random_uniform


def extrapolative_initial_design(X_lower, X_upper, is_env, task, N):

    # Create grid for the system size
    idx = is_env == 1
    X_upper_re = np.exp(task.retransform(X_upper))

    g = np.array([X_upper_re[idx] / float(i) for i in [4, 8, 16, 32]])[:, 0]
    g = np.true_divide((np.log(g) - task.original_X_lower[idx]),
           (task.original_X_upper[idx] - task.original_X_lower[idx]))

    X = init_random_uniform(X_lower, X_upper, N)

    X[:, is_env == 1] = \
        np.tile(g, np.ceil(X.shape[0] / 4.))[:X.shape[0], np.newaxis]

    return X
