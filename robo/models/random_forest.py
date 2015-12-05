'''
Created on Oct 11, 2015

@author: Aaron Klein
'''

import numpy as np


from robo.models.base_model import BaseModel

import pyrfr.regression


class RandomForest(BaseModel):
    '''
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

    seed: int
        The seed that is passed to the random_forest_run library.
    '''

    def __init__(self, types, num_trees=30,
                 do_bootstrapping=True,
                 n_points_per_tree=0,
                 ratio_features=0.5,
                 min_samples_split=1,
                 min_samples_leaf=1,
                 max_depth=20,
                 eps_purity=1e-8,
                 max_num_nodes=1000,
                 seed=42):

        self.types = types
        self.types.dtype = np.uint

        self.rf = pyrfr.regression.binary_rss()
        self.rf.num_trees = num_trees
        self.rf.seed = seed
        self.rf.do_bootstrapping = do_bootstrapping
        self.rf.num_data_points_per_tree = n_points_per_tree
        self.rf.max_features = int(types.shape[0] * ratio_features)
        self.rf.min_samples_to_split = min_samples_split
        self.rf.min_samples_in_leaf = min_samples_leaf
        self.rf.max_depth = max_depth
        self.rf.epsilon_purity = eps_purity
        self.rf.max_num_nodes = max_num_nodes

        # This list well be read out by save_iteration() in the solver
        self.hypers = [num_trees, max_num_nodes, do_bootstrapping,
                       n_points_per_tree, ratio_features, min_samples_split,
                       min_samples_leaf, max_depth, eps_purity, seed]
        self.seed = seed

    def train(self, X, Y, **kwargs):
        '''
        Trains the random forest on X and Y.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, 1)
            The corresponding target values.
        '''

        self.X = X
        self.Y = Y

        data = pyrfr.regression.numpy_data_container(self.X,
                                                      self.Y[:, 0],
                                                      self.types)

        self.rf.fit(data)

    def predict(self, Xtest, **kwargs):
        """
        Returns the predictive mean and variance of the objective function
        at X average over the predictions of all trees.

        Parameters
        ----------
        Xtest: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(1,)
            predictive mean
        np.array(1,)
            predictive variance

        """
        mean = np.zeros(Xtest.shape[0])
        var = np.zeros(Xtest.shape[0])

        # TODO: Would be nice if the random forest supports batch predictions
        for i, x in enumerate(Xtest):
            mean[i], var[i] = self.rf.predict(x)

        return mean[:, np.newaxis], var[:, np.newaxis]

    def predict_each_tree(self, Xtest, **args):
        pass

    def update(self, X, y):
        pass


class RandomForestWithInstances(RandomForest):
    '''
    Interface to the random forest that takes instance features
    into account.

    Parameters
    ----------
    types: np.ndarray (D)
        Specifies the number of categorical values of an input dimension. Where
        the i-th entry corresponds to the i-th input dimension. Let say we have
        2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass np.array([2, 0]). Note that we count starting from 0.
    instance_features: np.ndarray (I, K)
        Contains the K dimensional instance features
        of the I different instances
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

    seed: int
        The seed that is passed to the random_forest_run library.
    '''

    def __init__(self, types,
                 instance_features,
                 num_trees=30,
                 do_bootstrapping=True,
                 n_points_per_tree=0,
                 ratio_features=0.5,
                 min_samples_split=1,
                 min_samples_leaf=1,
                 max_depth=20,
                 eps_purity=1e-8,
                 max_num_nodes=1000,
                 seed=42):

        self.instance_features = instance_features
        super(RandomForestWithInstances, self).__init__(types,
                                                        num_trees,
                                                        do_bootstrapping,
                                                        n_points_per_tree,
                                                        ratio_features,
                                                        min_samples_split,
                                                        min_samples_leaf,
                                                        max_depth,
                                                        eps_purity,
                                                        max_num_nodes,
                                                        seed)

    def predict(self, Xtest):
        """
        Returns the predictive mean and variance marginalised over all
        instances.

        Parameters
        ----------
        Xtest: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(1,)
            predictive mean over all instances
        np.array(1,)
            predictive variance over all instances
        """
        X_ = np.repeat(Xtest, self.instance_features.shape[0], axis=0)
        I_ = np.tile(self.instance_features, (Xtest.shape[0], 1))

        X_ = np.concatenate((X_, I_), axis=1)

        mu, var = super(RandomForestWithInstances, self).predict(X_)

        return mu.mean(), var.mean()
