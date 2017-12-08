import gpflow
import tensorflow as tf


class LearningCurveKernel(gpflow.kernels.Kernel):

    def __init__(self, alpha=0.6, beta=.3, input_dim=1, active_dims=[0]):
        """
        Kernel from the the Freeze Thaw paper [1] to model learning curves.

        [1] Freeze-Thaw Bayesian Optimization
            Kevin Swersky and Jasper Snoek and Ryan P. Adams
        """

        super().__init__(input_dim=input_dim, active_dims=active_dims)
        self.alpha = gpflow.params.Parameter(alpha,
                                             prior=gpflow.priors.LogNormal(mu=0, var=1))
        self.beta = gpflow.params.Parameter(beta,
                                            prior=gpflow.priors.LogNormal(mu=0, var=1))

    @gpflow.decors.params_as_tensors
    def K(self, X, X2=None):
        X = tf.reshape(X[:, self.active_dims], [-1, 1])
        if X2 is None:
            X2 = X
        else:
            X2 = X2[:, self.active_dims]

        return tf.pow(self.beta, self.alpha) / tf.pow((X + tf.transpose(X2) + self.beta), self.alpha)

    @gpflow.decors.params_as_tensors
    def Kdiag(self, X):
        X = tf.reshape(X[:, self.active_dims], [-1, 1])
        return tf.reshape(tf.pow(self.beta, self.alpha) / tf.pow((2 * X + self.beta), self.alpha), (-1,))
