import unittest
import numpy as np

from robo.solver.random_search import RandomSearch
from robo.task.base_task import BaseTask


class TestRandomSearch(unittest.TestCase):

    def test_run(self):
        class ExampleTask(BaseTask):

            def __init__(self):
                X_lower = np.array([0, 0])
                X_upper = np.array([1, 1])
                super(ExampleTask, self).__init__(X_lower, X_upper)

            def objective_function(self, x):
                y = np.random.normal()
                return np.array([[y]])

        task = ExampleTask()

        # Shape matching run
        rs = RandomSearch(task=task)
        inc, inc_val = rs.run(10)
        assert len(inc.shape) == 2
        assert inc.shape[0] == 1
        assert inc.shape[1] == task.X_lower.shape[0]
        assert len(inc_val.shape) == 2
        assert inc_val.shape[0] == 1
        assert inc_val.shape[1] == 1
        # Check if point is in the bounds
        assert np.all(inc >= task.X_lower)
        assert np.all(inc <= task.X_upper)

        # Shape matching choose_next
        rs = RandomSearch(task=task)
        x_ = rs.choose_next()
        assert len(x_.shape) == 2
        assert x_.shape[0] == 1
        assert x_.shape[1] == task.X_lower.shape[0]

        # Check if point is in the bounds
        assert np.all(x_ >= task.X_lower)
        assert np.all(x_ <= task.X_upper)

if __name__ == "__main__":
    unittest.main()
