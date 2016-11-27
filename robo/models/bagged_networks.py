
import numpy as np


class BaggedNets(object):
    """
     Abstract base class for all models
    """

    def __init__(self, base_model_class, num_models=16,
                 bootstrap_size=None, bootstrap_with_replacement=True, **init_kwargs):
        """
        Parameters
        ----------
            base_model_class : custom_class with a fit and predict method
                base model fitted to bootstrapped samples of the data
            num_models : int
                number of base models fitted
            bootstrap_size : int
                size of the bootstrap sample
            bootstrap_with_replacement : boolean
                whether or not to draw samples from the data with or without replacement
            **kwargs:
                all remaining arguments are passed to base_model_class.__init__()
        """
        self.base_model_class = base_model_class
        self.init_kwargs = init_kwargs
        self.num_models = num_models
        self.bootstrap_size = bootstrap_size
        self.bootstrap_with_replacement = bootstrap_with_replacement
        self.models = None

    def train(self, X, Y, **kwargs):
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
        **kwargs:
            all other arguments will be passed to the training method of the base model
        """
        self.X = X
        self.Y = Y
        self.models = []
        self.bootstrap_indices = []

        bss = X.shape[0] if self.bootstrap_size is None else self.bootstrapsize

        for i in range(self.num_models):

            indices=np.random.choice(np.arange(X.shape[0]), size = bss, replace = self.bootstrap_with_replacement)
            self.bootstrap_indices.append(indices)

            self.models.append( self.base_model_class(X.shape[1], **self.init_kwargs) )
            self.models[-1].train(np.copy(X[indices]), np.copy(Y[indices]), **kwargs)

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

        predictions = [m.predict(X)[:, 0, 0] for m in self.models]

        m = np.mean(predictions, axis=0)
        v = np.var(predictions, axis=0)
        
        return m[:, np.newaxis], v[:, np.newaxis]

    def predictive_gradients(self, X):
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

        predictions = [m.predict_gradient(X)[:, 0, 0] for m in self.models]

        m = np.mean(predictions, axis=0)
        
        return m[:, np.newaxis]

    def predict_variance(self, X1, X2):
        raise NotImplementedError()

    def get_json_data(self):
        """
        Json getter function'

        :return: Dict object
        """
        jsonData = {'X': self.X if self.X is None else self.X.tolist(),
                    'Y': self.Y if self.Y is None else self.Y.tolist(),
                    'hyperparameters': " "}
        return jsonData

