import abc


class BaseAcquisitionFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        """
        A base class for acquisition_functions functions.

        Parameters
        ----------
        model : Model object
            Models the objective function.
        """
        self.model = model

    def update(self, model):
        """
        This method will be called if the model is updated. E.g.
        Entropy search uses it to update it's approximation of P(x=x_min)

        Parameters
        ----------
        model : Model object
            Models the objective function.
        """

        self.model = model

    def _multiple_inputs(foo):
        def wrapper(self, X, **kwargs):
            if len(X.shape) > 1:
                a = [foo(self, x, **kwargs) for x in X]
            else:
                a = foo(self, X, **kwargs)
            return a
        return wrapper

    @abc.abstractmethod
    def compute(self, x, derivative=False):
        """
        Computes the acquisition_functions value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        x: np.ndarray(D,), The input point where the acquisition_functions function
            should be evaluate.

        derivative: Boolean
            If is set to true also the derivative of the acquisition_functions
            function at X is returned
        """
        pass

    def __call__(self, x, **kwargs):
        return self.compute(x, **kwargs)

    def get_json_data(self):
        """
        Json getter function

        :return: Dict() object
        """

        json_data = dict()
        json_data = {"type": __name__}
        return json_data
