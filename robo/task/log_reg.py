import os
import numpy as np

from robo.task.base_task import BaseTask

from wrappingLogistic import logistic

class LogisticRegression(BaseTask):

    def __init__(self, path="/mhome/kleinaa/data/mnist_npy"):
        self.path = path
        X_lower = np.array([0, 0, 20, 5])
        X_upper = np.array([10, 1, 2000, 2000])
        super(LogisticRegression, self).__init__(X_lower, X_upper)
        self.train = np.load(os.path.join(self.path, "x_train.npy"))
        self.train_targets = np.load(os.path.join(self.path, "y_train.npy"))
        self.valid = np.load(os.path.join(self.path, "x_valid.npy"))
        self.valid_targets = np.load(os.path.join(self.path, "y_valid.npy"))
        
        self.train = np.array(self.train.reshape((-1, 784)), dtype=np.float32)
        self.valid = np.array(self.valid.reshape((-1, 784)), dtype=np.float32)

        # Use whole training data set        
        self.n_train = self.train.shape[0]

    def objective_function(self, x):
        params = dict()
        params["lrate"] = x[0, 0]
        params["l2_reg"] = x[0, 1]
        params["batchsize"] = x[0, 2]
        params["n_epochs"] = x[0, 3]

        kwargs = dict()
        kwargs['train'] = self.train[:self.n_train]
        kwargs['train_targets'] = self.train_targets[:self.n_train]
        kwargs['valid'] = self.valid
        kwargs['valid_targets'] = self.valid_targets
        y = logistic(params, **kwargs)

        if np.any(np.isinf(y)):
            y = 1
        return np.array([[y]])

    def objective_function_test(self, x):
        return self.objective_function(x)
#         self.test = np.load(os.path.join(self.path, "x_test.npy"))
#         self.test_targets = np.load(os.path.join(self.path, "y_test.npy"))
#         
#         self.test = np.array(self.valid.reshape((-1, 784)), dtype=np.float32)
#         params = dict()
#         params["lrate"] = x[0, 0]
#         params["l2_reg"] = x[0, 1]
#         params["batchsize"] = x[0, 2]
#         params["n_epochs"] = x[0, 3]
# 
#         # Train on validation + train data set and return performance on test data set 
#         kwargs = dict()
#         kwargs['train'] = np.concatenate((self.train, self.valid), axis=0)
#         kwargs['train_targets'] = np.concatenate((self.train_targets, self.valid_targets), axis=0)
#         #kwargs['train'] = self.train
#         #kwargs['train_targets'] = self.train_targets
#         kwargs['valid'] = self.test
#         kwargs['valid_targets'] = self.test_targets
#         y = logistic(params, **kwargs)
# 
#         if np.any(np.isinf(y)):
#             y = np.array([[1]])
#         return np.array([[y]])
        