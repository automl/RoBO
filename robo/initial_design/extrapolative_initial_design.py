'''
Created on Dec 21, 2015

@author: Aaron Klein
'''

import numpy as np

from robo.initial_design.init_random_uniform import init_random_uniform


def extrapolative_initial_design(task, N):

    # Index of the environmental variable
    idx = task.is_env == 1
    # Upper bound of the dataset size on a linear scale
    X_upper_re = np.exp(task.retransform(task.X_upper))[idx]
    # Compute the dataset size on a linear scale for a 1/4, 1/8, 1/16 and 1/32 of the data
    s = np.array([X_upper_re / float(i) for i in [4, 8, 16, 32]])[:, 0]
    log_s = np.log(s)

    # Transform it back to [0, 1] space
    s = np.true_divide((log_s - task.original_X_lower[idx]),
           (task.original_X_upper[idx] - task.original_X_lower[idx]))

    # Draw random points in the configuration space and evaluate them on the predifined data subsets
    X = init_random_uniform(task.X_lower, task.X_upper, N)
    X[:, task.is_env == 1] = \
        np.tile(s, np.ceil(X.shape[0] / 4.))[:X.shape[0], np.newaxis]

    return X
