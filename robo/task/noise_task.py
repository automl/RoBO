'''
Created on 28.10.2015

@author: Lukas Voegtle
'''
import numpy as np

from robo.task.base_task import BaseTask


class NoiseTask(BaseTask):
    """
    Class for a task which adds normal distributed noise to another task
    """

    def __init__(self, base_task, noise_scale=1):
        """
        Creates a task which wraps another task and adds normal distributed noise.


        :param base_task: The base task
        :param noise_scale: noise standard deviation
        """
        self.noise_scale = noise_scale
        self.base_task = base_task
        super(NoiseTask, self).__init__(base_task.X_lower, base_task.X_upper, base_task.opt, base_task.fopt, base_task.do_scaling)

    def objective_function(self, x):
        res = self.base_task.objective_function(x)
        res += np.random.normal(0, self.noise_scale, res.shape)

        return res

    def objective_function_test(self, x):
        res = self.base_task.objective_function_test(x)
        res += np.random.normal(0, self.noise_scale, res.shape)

        return res