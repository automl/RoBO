
import numpy as np


from robo.models.base_model import BaseModel

try:
    import pyrfr.regression as reg
except:
    raise ValueError("If you want to use Random Forests you have to install the following dependencies:\n"
                     "Pyrfr (pip install pyrfr)")


class RandomForest(BaseModel):

    def __init__(self, num_trees=30,
                 do_bootstrapping=True,
                 n_points_per_tree=0,
                 compute_oob_error=False,
                 return_total_variance=True,
                 rng=None):
        """
        Interface for the random_forest_run library to model the
        objective function with a random forest.

        Parameters
        ----------
        num_trees: int
            The number of trees in the random forest.
        do_bootstrapping: bool
            Turns on / off bootstrapping in the random forest.
        n_points_per_tree: int
            Number of data point per tree. If set to 0 then we will use all data points in each tree
        compute_oob_error: bool
            Turns on / off calculation of out-of-bag error. Default: False
        return_total_variance: bool
            Return law of total variance (mean of variances + variance of means, if True)
            or explained variance (variance of means, if False). Default: True
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.reg_rng = reg.default_random_engine(self.rng.randint(1000))

        self.n_points_per_tree = n_points_per_tree

        self.rf = reg.binary_rss_forest()
        self.rf.options.num_trees = num_trees
        self.rf.options.do_bootstrapping = do_bootstrapping
        self.rf.options.num_data_points_per_tree = n_points_per_tree
        self.rf.options.compute_oob_error = compute_oob_error
        self.rf.options.compute_law_of_total_variance = return_total_variance

    def train(self, X, y, **kwargs):
        """
        Trains the random forest on X and y.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        """

        self.X = X
        self.y = y

        if self.n_points_per_tree == 0:
            self.rf.options.num_data_points_per_tree = X.shape[0]

        data = reg.default_data_container(self.X.shape[1])

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)

        self.rf.fit(data, self.reg_rng)

    def predict(self, X_test, **kwargs):
        """
        Returns the predictive mean and variance of the objective function
        at X average over the predictions of all trees.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(1,)
            predictive mean
        np.array(1,)
            predictive variance

        """
        mean = np.zeros(X_test.shape[0])
        var = np.zeros(X_test.shape[0])

        for i, x in enumerate(X_test):
            mean[i], var[i] = self.rf.predict_mean_var(x)

        return mean, var

    def predict_each_tree(self, X_test, **args):
        pass

    def sample_functions(self, X_test, n_funcs=1):
        pass

    def __getstate__(self):
        sdict = self.__dict__.copy()
        del sdict['reg_rng']  # delete not-pickleable objects
        return sdict

    def __setstate__(self, sdict):
         self.__dict__.update(sdict)
         self.reg_rng = reg.default_random_engine(sdict['rng'].randint(1000))
