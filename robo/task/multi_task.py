# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:10:34 2015

@author: aaron
"""

import numpy as np

from robo.task.base_task import BaseTask
from robo.task.branin import Branin


class MultiTaskBranin(BaseTask):
    '''
    classdocs
    '''

    def __init__(self):
        self.branin = Branin
        self.shifted_branin = ShiftedBranin
        
        # Add dimension for the tasks
        X_lower = np.concatenate((self.original_task.original_X_lower,
                                  np.array([0])))

        X_upper = np.concatenate((self.original_task.original_X_upper,
                                  np.array([1])))        
        
        super(MultiTaskBranin, self).__init__(X_lower, X_upper,
                self.branin.opt, self.branin.fopt)
        
        
    def objective_function(self, x):
        # Evaluate shifted branin
        if x[0, -1] == 0: 
            return self.shifted_branin.objective_function(x[:, :-1])
        # Evaluate true branin        
        elif x[0, -1] == 1:
            return self.branin.objective_function(x[:, :-1])

    def objective_function_test(self, x):
        # Evaluate shifted branin
        if x[0, -1] == 0: 
            return self.shifted_branin.objective_function_test(x[:, :-1])
        # Evaluate true branin        
        elif x[0, -1] == 1:
            return self.branin.objective_function_test(x[:, :-1])

