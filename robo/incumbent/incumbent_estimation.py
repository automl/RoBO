

class IncumbentEstimation(object):

    def __init__(self, model, X_lower, X_upper):
        """
        A base class to estimate the global optimizer aka incumbent.

        Parameters
        ----------
        model : Model object
            Models the objective function.
        X_lower : (D) numpy array
            Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        X_upper : (D) numpy array
            Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        """
        self.model = model
        self.X_upper = X_upper
        self.X_lower = X_lower

    def estimate_incumbent(self, startpoints):
        """
        Abstract function that estimates the current incumbent
        by starting one local search from each of the startpoints.

        Parameters
        ----------
        startpoints : (N, D) numpy array
            In the case of local search, we start form each point a
            separated local search procedure

        Returns
        -------
        np.ndarray(1, D)
            Incumbent
        np.ndarray(1,1)
            Incumbent value
        """

        pass
