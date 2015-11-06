'''
Created on Jul 21, 2015

@author: Aaron Klein
'''
import logging

import numpy as np

from robo.task.base_task import BaseTask

logger = logging.getLogger(__name__)


class REMBO(BaseTask):
    '''
    classdocs
    '''

    def __init__(self, X_lower, X_upper, d, embedding="Gauss"):
        # Dimensions of the original space
        self.d_orig = X_lower.shape[0]
        # Dimension of the embedded space
        self.d = d
        # Create random embedding
        if embedding == 'Gauss':
            A = np.sqrt(self.d) * np.random.normal(0.0, 1.0, (self.d_orig, self.d))
        elif embedding == 'diagonal':
            indic = np.random.choice(range(self.d), self.d_orig)
            eye = np.eye(d)
            A = np.dot(np.diag(np.random.choice([-1.0, 1.0], self.d_orig)), eye[indic])
        elif embedding == 'orthogonal':
            A = np.random.normal(0.0, 1.0, (self.d_orig, self.d))
            U, s, V = np.linalg.svd(A, full_matrices=False)
            A = np.dot(U, V)
            A = np.sqrt(self.d_orig) * A
        elif embedding == 'identity':
            A = np.zeros((self.d_orig, self.d))
            for i in xrange(self.d):
                A[i, i] = 1.0
        else:
            logger.error('ERROR: Unknown embedding option: ' + str(embedding))
            return
        self.A = A

        self.original_X_lower = X_lower
        self.original_X_upper = X_upper

        # Scale the original space to [-1, 1]
        self.original_scaled_X_lower = -1 * np.ones([self.d_orig])
        self.original_scaled_X_upper = 1 * np.ones([self.d_orig])
        # The embedded configuration space
        super(REMBO, self).__init__(- np.sqrt(self.d) * np.ones(self.d),
                                    np.sqrt(self.d) * np.ones(self.d), do_scaling=False)

    def evaluate(self, x):
        # Project to original space
        x_transform = np.array([np.dot(self.A, e) for e in x])
        # Convex projection
        x_projected = np.fmax(
            self.original_scaled_X_lower, np.fmin(
                self.original_scaled_X_upper, x_transform))
        # Rescale back to original space
        x_rescaled = (self.original_X_upper - self.original_X_lower) * (x_projected - self.original_scaled_X_lower) / \
            (self.original_scaled_X_upper - self.original_scaled_X_lower) + self.original_X_lower
        return self.objective_function(x_rescaled)
