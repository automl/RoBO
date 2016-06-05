
import numpy as np


class BaseModel(object):
    """
     Abstract base class for all models
    """

    def __init__(self, *args, **kwargs):
        self.X = None
        self.Y = None

    def train(self, X, Y):
        """
        Trains the model on the provided data.
            
        Parameters
        ----------
        X: np.ndarray (N, D)
            Input datapoints. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, T)
            The corresponding target values.
            The dimensionality of Y is (N, T), where N has to 
            match the number of points of X and T is the number of objectives
        """
        self.X = X
        self.Y = Y

    def update(self, X, Y):
        X = np.append(self.X, X, axis=0)
        Y = np.append(self.Y, Y, axis=0)
        self.train(X, Y)

    def predict(self, X):
        """
        Predicts for a given X matrix the target values
        
        Parameters
        ----------
        X: np.ndarray (N, D)
            Test datapoints. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
            
        Returns
        ----------
            The mean and variance of the test datapoint.
        """
        raise NotImplementedError()

    def predict_variance(self, X1, X2):
        raise NotImplementedError()

    def predictive_gradients(self, X=None):
        """
        Calculates the predictive gradients (gradient of the prediction)
        
        Parameters
        ----------
        
        X: np.ndarray (N, D)
            The points to predict the gradient for
        Returns
        ----------
            The gradients at X
        """
        raise NotImplementedError()

    def get_json_data(self):
        """
        Json getter function'

        :return: Dict object
        """
        jsonData = dict()
        jsonData = {'X': self.X, 'Y':self.Y, 'hyperparameters':" " }
        return jsonData

