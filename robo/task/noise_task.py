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
        if base_task.do_scaling:
            super(NoiseTask, self).__init__(base_task.original_X_lower, base_task.original_X_upper,
                                            base_task.original_opt, base_task.original_fopt, base_task.do_scaling)
        else:
            super(NoiseTask, self).__init__(base_task.X_lower, base_task.X_upper,
                                            base_task.opt, base_task.fopt, base_task.do_scaling)

    def objective_function(self, x, rng = None):
        """
        Objective function delegates to the base task and adds noise.
        Parameters
        ----------
        seed: int
            Number that is passed to the numpy random number generator

        """
        if rng is None:
            rng = np.random.RandomState(42)
        res = self.base_task.objective_function(x)
        res += rng.normal(0, self.noise_scale, res.shape)

        return res

    def objective_function_test(self, x):
        """
        Objective test function delegates to the base task (without noise).
        """
        return self.base_task.objective_function_test(x)
