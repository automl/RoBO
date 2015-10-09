import numpy as np

from robo.task.base_task import BaseTask
from robo.task.log_reg import LogisticRegression

class EnvLogisticRegression(BaseTask):

    def __init__(self, path="/mhome/kleinaa/data/mnist_npy"):
        self.log_reg = LogisticRegression(path)
        
        X_lower = np.concatenate((self.log_reg.original_X_lower, np.array([6.91])))
        X_upper = np.concatenate((self.log_reg.original_X_upper, np.array([10.81978])))
        self.is_env = np.zeros([self.log_reg.n_dims])
        self.is_env = np.concatenate((self.is_env, np.array([1])))
        
        super(EnvLogisticRegression, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        shuffle = np.random.permutation(np.arange(self.log_reg.train.shape[0]))
        self.log_reg.train, self.log_reg.train_targets = self.log_reg.train[shuffle], self.log_reg.train_targets[shuffle]
        self.log_reg.n_train = int(np.exp(x[0, -1]))

        y = self.log_reg.objective_function(x)

        return y

    def objective_function_test(self, x):
        return self.log_reg.objective_function_test(x[:, :-1])
    