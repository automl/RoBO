import numpy as np

from robo.task.base_task import BaseTask


class Hartmann3(BaseTask):

    def __init__(self):

        X_lower = np.array([0, 0, 0])
        X_upper = np.array([1, 1, 1])
        opt = np.array([[0.114614, 0.555649, 0.852547]])
        fopt = np.array([[-3.86278]])

        super(Hartmann3, self).__init__(X_lower, X_upper, opt, fopt)

        self.alpha = [1.0, 1.2, 3.0, 3.2]
        self.A = np.array([[3.0, 10.0, 30.0],
                           [0.1, 10.0, 35.0],
                           [3.0, 10.0, 30.0],
                           [0.1, 10.0, 35.0]])
        self.P = 0.0001 * np.array([[3689, 1170, 2673],
                                    [4699, 4387, 7470],
                                    [1090, 8732, 5547],
                                    [381, 5743, 8828]])

    def objective_function(self, x):

        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(3):
                internal_sum = internal_sum \
                            + self.A[i, j] * (x[:, j] \
                            - self.P[i, j]) ** 2
            external_sum = external_sum + self.alpha[i] * np.exp(-internal_sum)

        return -external_sum[:, np.newaxis]

    def objective_function_test(self, x):

        return self.objective_function(x)
