import numpy as np

from robo.task.base_task import BaseTask

from wrappingLogistic import logistic
from sklearn.datasets import load_digits

class LogisticRegression(BaseTask):

    def __init__(self):
        X_lower = np.array([0, 0, 20, 5])
        X_upper = np.array([10, 1, 2000, 2000])
        super(LogisticRegression, self).__init__(X_lower, X_upper)

        self.train = np.load('/home/aaron/data/mnist_train.npy')
        self.train_targets = np.load('/home/aaron/data/mnist_train_targets.npy')
        self.valid = np.load('/home/aaron/data/mnist_valid.npy')
        self.valid_targets = np.load('/home/aaron/data/mnist_valid_targets.npy')
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

        return np.array([[y]])

    def evaluate_test(self, x):
        self.test = np.load('/home/aaron/data/mnist_test.npy')
        self.test_targets = np.load('/home/aaron/data/mnist_train_test.npy')
        
        params = dict()
        params["lrate"] = x[0, 0]
        params["l2_reg"] = x[0, 1]
        params["batchsize"] = x[0, 2]
        params["n_epochs"] = x[0, 3]

        kwargs = dict()
        kwargs['train'] = np.concatenate((self.train, self.valid), axis=0)
        kwargs['train_targets'] = np.concatenate((self.train_targets, self.valid_targets), axis=0)
        kwargs['valid'] = self.test
        kwargs['valid_targets'] = self.test_targets
        y = logistic(params, **kwargs)

        return np.array([[y]])
        