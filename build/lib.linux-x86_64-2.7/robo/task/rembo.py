import logging
import numpy as np

from robo.task.base_task import BaseTask

logger = logging.getLogger(__name__)


class REMBO(BaseTask):

    def __init__(self, X_lower, X_upper, d):
        """
        Random EMbedding Bayesian Optimization [1] maps the original space
        in a lower dimensional subspace via a random embedding matrix and
        performs Bayesian Optimization only in this lower dimensional
        subspace.

        [1] Ziyu Wang and Masrour Zoghi and Frank Hutter and David Matheson
            and Nando de Freitas
            Bayesian Optimization in High Dimensions via Random Embeddings
            In: Proceedings of the 23rd international joint conference
            on Artificial Intelligence (IJCAI)

        Parameters
        ----------
        X_lower : (D,) numpy array
            The lower bound of the input space.
        X_upper: (D,) numpy array
            The upper bound of the input space.
        d: int
            Number of dimensions for the lower dimensional subspace
        """

        # Dimensions of the original space
        self.d_orig = X_lower.shape[0]

        # Dimension of the embedded space
        self.d = d

        # Draw random matrix from a normal distribution
        self.A = np.sqrt(self.d) * np.random.normal(loc=0.0,
                                               scale=1.0,
                                               size=(self.d_orig, self.d))
        # Save original space
        self.original_X_lower = X_lower
        self.original_X_upper = X_upper

        # Scale the original space to [-1, 1]
        self.original_scaled_X_lower = -1 * np.ones([self.d_orig])
        self.original_scaled_X_upper = 1 * np.ones([self.d_orig])

        # The embedded configuration space
        super(REMBO, self).__init__(X_lower=-np.sqrt(self.d) * np.ones(self.d),
                                    X_upper=np.sqrt(self.d) * np.ones(self.d),
                                    do_scaling=False)

    def evaluate(self, x):

        # Project to original space
        x_transform = np.array([np.dot(self.A, e) for e in x])

        # Convex projection
        x_projected = np.fmax(self.original_scaled_X_lower,
                              np.fmin(self.original_scaled_X_upper,
                                      x_transform))

        # Rescale back to original space
        x_rescaled = (self.original_X_upper - self.original_X_lower) \
             * (x_projected - self.original_scaled_X_lower) / \
             (self.original_scaled_X_upper - self.original_scaled_X_lower) \
             + self.original_X_lower

        return self.objective_function(x_rescaled)
