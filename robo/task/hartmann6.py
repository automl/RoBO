'''
Created on 14.07.2015

@author: Aaron Klein
'''

import numpy as np

from robo.task.base_task import BaseTask


class Hartmann6(BaseTask):
    '''
    classdocs
    '''

    def __init__(self):
        self.X_lower = np.array([0, 0, 0, 0, 0, 0])
        self.X_upper = np.array([1, 1, 1, 1, 1, 1])
        self.n_dims = 6
        self.opt = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
        self.fopt = np.array([[-3.32237]])

        self.alpha = [1.00, 1.20, 3.00, 3.20]
        self.A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                      [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                      [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                      [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        self.P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])

    def objective_function(self, x):
        """6d Hartmann test function
            input bounds:  0 <= xi <= 1, i = 1..6
            global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
            min function value = -3.32237
        """

        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum = internal_sum + self.A[i, j] * (x[:, j] - self.P[i, j]) ** 2
            external_sum = external_sum + self.alpha[i] * np.exp(-internal_sum)

        return -external_sum[:, np.newaxis]
