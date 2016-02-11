import numpy as np


class BaseTask(object):

    def __init__(self, X_lower, X_upper,
        opt=None, fopt=None,
        types=None, do_scaling=True):
        """
        Defines the interface of tasks. A task contains a function
        handle that calls the objective function and and the input
        bounds. Furthermore it can contain additional task specific
        information such as the location and the function value
        of the global optima. New tasks should be derived from
        this base class.

        Parameters
        ----------
        X_lower : (D,) numpy array
            The lower bound of the input space.
        X_upper: (D,) numpy array
            The upper bound of the input space.
        opt: (N, D) numpy array
            The global optima of the objective function (if known).
            Allows to compute and plot the distance of the incumbent
            to the global optimum.
        fopt: (N, 1) numpy array
            Function value of the N global optima (if known). Useful
            to compute the immediate or cumulative regret.
        types: (D, ) numpy array
            It specifies the number of categorical values
            of each dimension:
            0 if the dimension is continuous.
            2,3,.. if the dimension consists of 2,3,.. categorical
            values.
            This is only needed for the random forest. In the case
            you use only GPs set it to None
        do_scaling: boolean
            If set to true the input space is scaled to [0, 1]. Useful
            to specify priors for the kernel lengthscale.
        """

        self.X_lower = X_lower
        self.X_upper = X_upper
        self.n_dims = self.X_lower.shape[0]

        assert self.n_dims == self.X_upper.shape[0]

        if types is None:
            self.types = np.zeros([self.n_dims], dtype=np.uint)
        else:
            self.types = np.array(types, dtype=np.uint)

        self.opt = opt
        self.fopt = fopt
        self.do_scaling = do_scaling

        if do_scaling:
            self.original_X_lower = self.X_lower
            self.original_X_upper = self.X_upper
            self.original_opt = opt
            self.original_fopt = fopt

            self.X_lower = np.zeros(self.original_X_lower.shape)
            self.X_upper = np.ones(self.original_X_upper.shape)
        else:
            self.X_lower = self.X_lower
            self.X_upper = self.X_upper

        if self.opt is not None:
            self.opt = self.transform(self.opt)

    def objective_function(self, x):
        """
        If you derive from this class make sure to override
        this function by a function that calls your objective
        function.

        Parameters
        ----------
        X: np.ndarray (1, D)
            Data point where the objective function should
            be evaluate.

        Returns
        ----------
        np.ndarray (1, 1)
            Function value of the objective function at x
        """
        pass

    def objective_function_test(self, x):
        """
        If there is a different objective function for offline testing, e.g
        testing a machine learning on a hold extra test set instead
        on a validation set override this function here.

        Parameters
        ----------
        X: np.ndarray (1, D)
            Data point where the objective function should
            be evaluate.

        Returns
        ----------
        np.ndarray (1, 1)
            Function value of the objective function at x
        """
        pass

    def transform(self, x):
        """
        Transforms from  original space to the space [0, 1]
        Parameters
        ----------
        X: np.ndarray (1, D)
            Data point in original space

        Returns
        ----------
        np.ndarray (1, D)
            Input point in [0, 1] input space
        """

        return np.true_divide((x - self.original_X_lower),
           (self.original_X_upper - self.original_X_lower))

    def retransform(self, x):
        """
        Scales from [0, 1] back to original space

        Parameters
        ----------
        X: np.ndarray (1, D)
            Data point in [0, 1] space

        Returns
        ----------
        np.ndarray (1, D)
            Input point in original input space
        """

        return (self.original_X_lower + (self.original_X_upper - self.original_X_lower) * x)

    def evaluate(self, x):
        """
        Wrapper function of the objective function that is used
        inside the solver to evaluate x. It rescales x from [0,1]
        to the original space and makes sure the x inside the
        bounds.

        Parameters
        ----------
        X: np.ndarray (1, D)
            Data point where the objective function should
            be evaluate.

        Returns
        ----------
        np.ndarray (1, 1)
            Function value of the objective function at x
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_dims
        assert np.all(x >= self.X_lower)
        assert np.all(x <= self.X_upper)

        if self.do_scaling:
            x = self.retransform(x)
        return (self.objective_function(x))

    def evaluate_test(self, x):
        """
        Wrapper function of the test objective function that can
        be used for an offline testing of incumbents.
        It rescales x from [0,1] to the original space and makes
        sure the x inside the bounds.

        Parameters
        ----------
        X: np.ndarray (1, D)
               Data point where the objective function should
            be evaluate.

        Returns
        ----------
        np.ndarray (1, 1)
            Function value of the objective function at x
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_dims
        assert np.all(x >= self.X_lower)
        assert np.all(x <= self.X_upper)

        if self.do_scaling:
            x = self.retransform(x)
        return self.objective_function_test(x)
