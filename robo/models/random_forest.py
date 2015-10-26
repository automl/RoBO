'''
Created on Oct 11, 2015

@author: Aaron Klein
'''

import numpy as np


import rfr


class RandomForest(object):
    '''
    classdocs
    '''


    def __init__(self, types, num_trees=30,
                 do_bootstrapping=True,
                 n_points_per_tree=0,
                 ratio_features=0.5,
                 min_samples_split=10,
                 min_samples_leaf=10,
                 max_depth=0,
                 eps_purity=1e-8,
                 seed=42):
        '''
        Constructor
        '''
        self.types = types
        
        self.num_trees = num_trees 
        self.do_bootstrapping = do_bootstrapping
        self.n_points_per_tree = n_points_per_tree 
        self.ratio_features = ratio_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.eps_purity = eps_purity
        self.seed = seed
    
    def train(self, X, Y):
        
        self.X = X
        self.Y = Y

        data = rfr.data_container.numpy_data_container_regression(self.X, self.Y, self.types)

        self.rf = rfr.regression.binary_rss()
        self.rf.num_trees = self.num_trees
        self.rf.seed = self.seed
        self.rf.do_bootstrapping = self.do_bootstrapping
        self.rf.num_data_points_per_tree= self.n_points_per_tree
        self.rf.max_features_per_split = int(X.shape[1] * self.ratio_features)
        self.rf.min_samples_to_split = self.min_samples_split
        self.rf.min_samples_in_leaf = self.min_samples_leaf 
        self.rf.max_depth = self.max_depth
        self.rf.epsilon_purity = self.eps_purity
        self.rf.fit(data)
    
    def predict(self, Xtest, **args):
        mean = np.zeros(Xtest.shape[0])
        var= np.zeros(Xtest.shape[0])
               
        # TODO: Would be nice if the random forest would support batch predictions
        for i, x in enumerate(Xtest):
            mean[i], var[i] =  self.rf.predict(x)

        return mean, var
    
    def predict_each_tree(self, Xtest, **args):
        pass
    
    def update(self, X, y):
        pass
        