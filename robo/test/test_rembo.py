import unittest
import numpy as np

from robo.task.rembo import REMBO
from robo.initial_design.init_random_uniform import init_random_uniform


class Test(unittest.TestCase):

    def test_rembo(self):

        class TestTask(REMBO):

            def __init__(self):
                X_lower = np.array([-5, 0])
                X_upper = np.array([10, 15])
                super(TestTask, self).__init__(X_lower, X_upper, d=1)

            def objective_function(self, x):
                return x

        t = TestTask()

        x = init_random_uniform(t.X_lower, t.X_lower, N=100)

        projected_scaled_x = t.evaluate(x)

        assert len(projected_scaled_x.shape) == 2
        assert projected_scaled_x.shape[1] == t.d_orig
        assert np.all(projected_scaled_x >= t.original_X_lower)
        assert np.all(projected_scaled_x <= t.original_X_upper)


if __name__ == "__main__":
    unittest.main()
