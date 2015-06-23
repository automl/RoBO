

class BaseModel(object):
    """
     Abstract base class for all models
    """
    def __init__(self, *args, **kwargs):
        self.X = None
        self.y = None

    def train(self, X, y):
        """
            Trains the model on the provided data.
            :param X: Input datapoints. The dimensionality of X is (N, D), with N as the number of points and D is the number of features.
            :type X: np.ndarray (N, D)
            :param Y: The corresponding target values. The dimensionality of Y is (N), where N has to match the number of points of X
            :type Y: np.ndarray (N)
        """
        self.X = X
        self.y = y

    def predict(self, X):
        """
            Predicts for a given X matrix the target values
            :param X: Test datapoints. The dimensionality of X is (N, D), with N as the number of points and D is the number of features.
            :type X: np.ndarray (N, D)
            :return The mean and variance of the testdatapoints.
        """
        raise NotImplementedError()

    def predict_variance(self, X1, X2):
        raise NotImplementedError()

    def predictive_gradients(self, Xnew, X=None):
        """
        Calculates the predictive gradients (gradient of the prediction)
        :param Xnew: The points to predict the gradient for
        :param X: TODO: Not implemented yet
        :return: Gradients(?)
        """
        raise NotImplementedError()
