import unittest
import numpy as np

from robo.util import incumbent_estimation


class TestIncumbentEstimation(unittest.TestCase):

    def test_projected_incumbent_estimation(self):
        X = np.random.randn(20, 10)
        y = np.sinc(X).sum(axis=1)

        class DemoModel(object):
            def train(self, X, y):
                self.X = X
                self.y = y

            def predict(self, X):
                return self.y, np.ones(self.y.shape[0])

        model = DemoModel()
        model.train(X, y)
        inc, inc_val = incumbent_estimation.projected_incumbent_estimation(model, X, proj_value=1)
        b = np.argmin(y)

        assert inc[-1] == 1
        assert np.all(inc[:-1] == X[b])
        assert inc_val == y[b]

if __name__ == "__main__":
    unittest.main()

