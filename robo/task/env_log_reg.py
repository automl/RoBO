import numpy as np

from robo.task.log_reg import LogisticRegression

class EnvLogisticRegression(LogisticRegression):

    def __init__(self):
        super(EnvLogisticRegression, self).__init__()
        self.X_lower = np.concatenate((self.X_lower, np.array([6.91])))
        self.X_upper = np.concatenate((self.X_upper, np.array([10.81978])))
        self.is_env = np.zeros([self.n_dims])
        self.is_env = np.concatenate((self.is_env, np.array([1])))
        self.n_dims = self.n_dims + 1

    def objective_function(self, x):
        shuffle = np.random.permutation(np.arange(self.train.shape[0]))
        self.train, self.train_targets = self.train[shuffle], self.train_targets[shuffle]
        self.n_train = int(np.exp(x[0, -1]))

        #eps = np.random.exponential(x[0, -1])
        y = super(EnvLogisticRegression, self).objective_function(x)

        return y

    def evaluate_test(self, x):
        return self.objective_function(x[:, :-1])