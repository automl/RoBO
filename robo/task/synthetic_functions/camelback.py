import numpy as np

from robo.task.base_task import BaseTask


class Camelback(BaseTask):

    def __init__(self):
        X_lower = np.array([-3, -2])
        X_upper = np.array([3, 2])
        super(Camelback, self).__init__(X_lower, X_upper)

    def objective_function(self, x):
        y = (4-2.1*(x[:, 0]**2)+((x[:, 0]**4)/3))*(x[:, 0]**2)+ x[:, 0]*x[:, 1]+(-4+4*(x[:, 1]**2))*(x[:, 1]**2)
        return y[:, np.newaxis]

    def objective_function_test(self, x):
        return self.objective_function(x)
