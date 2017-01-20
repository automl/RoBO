
import numpy as np


from robo.models.base_model import BaseModel

try:
    import pyrfr.regression
except:
    raise ValueError("If you want to use Random Forests you have to install the following dependencies:\n"
                     "Pyrfr (pip install pyrfr)")


class RandomForest(BaseModel):
    """
    Interface for the random_forest_run library to model the
    objective function with a random forest.

    Parameters
    ----------
    types: np.ndarray (D)
        Specifies the number of categorical values of an input dimension. Where
        the i-th entry corresponds to the i-th input dimension. Let say we have
        2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass np.array([2, 0]). Note that we count starting from 0.
    num_trees: int
        The number of trees in the random forest.
    do_bootstrapping: bool
        Turns on / off bootstrapping in the random forest.
    ratio_features: float
        The ratio of features that are considered for splitting.
    min_samples_split: int
        The minimum number of data points to perform a split.
    min_samples_leaf: int
        The minimum number of data points in a leaf.
    max_depth: int

    eps_purity: float

    max_num_nodes: int

    """

    def __init__(self, types, num_trees=30,
                 do_bootstrapping=True,
                 n_points_per_tree=0,
                 ratio_features=0.5,
                 min_samples_split=2,
                 min_samples_leaf=2,
                 max_depth=100,
                 eps_purity=1e-8,
                 max_num_nodes=0,
                 rng=None):

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # make sure types are uint
        self.types = np.array(types, dtype=np.uint)

        self.rf = pyrfr.regression.binary_rss()
        self.rf.num_trees = num_trees

        self.rf.do_bootstrapping = do_bootstrapping
        self.rf.num_data_points_per_tree = n_points_per_tree
        self.rf.max_features = int(types.shape[0] * ratio_features)
        self.rf.min_samples_to_split = min_samples_split
        self.rf.min_samples_in_leaf = min_samples_leaf
        self.rf.max_depth = max_depth
        self.rf.epsilon_purity = eps_purity
        self.rf.max_num_nodes = max_num_nodes

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

        data = pyrfr.regression.numpy_data_container(self.X,
                                                     self.y,
                                                     self.types)

        self.rf.fit(data)

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

        # TODO: Would be nice if the random forest supports batch predictions
        for i, x in enumerate(X_test):
            mean[i], var[i] = self.rf.predict(x)

        return mean, var

    def predict_each_tree(self, X_test, **args):
        pass

    def sample_functions(self, X_test, n_funcs=1):
        pass
