import unittest
import numpy as np

from robo.models.wrapper_bohamiann import WrapperBohamiann
from robo.models.wrapper_bohamiann import WrapperBohamiannMultiTask


class TestWrapperBohamiann(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        self.model = WrapperBohamiann()
        self.model.train(self.X, self.y)

    def test_predict(self):
        X_test = np.random.rand(10, 2)

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_get_incumbent(self):
        inc, inc_val = self.model.get_incumbent()

        b = np.argmin(self.y)
        np.testing.assert_almost_equal(inc, self.X[b], decimal=5)


class TestWrapperBohamiannMultiTask(unittest.TestCase):

    def setUp(self):
        X_task_1 = np.random.rand(10, 2)
        y_task_1 = np.sinc(X_task_1 * 10 - 5).sum(axis=1)
        X_task_1 = np.concatenate((X_task_1, np.zeros([10, 1])), axis=1)

        X_task_2 = np.random.rand(10, 2)
        y_task_2 = np.sinc(X_task_2 * 2 - 4).sum(axis=1)
        X_task_2 = np.concatenate((X_task_2, np.ones([10, 1])), axis=1)

        X_task_3 = np.random.rand(10, 2)
        y_task_3 = np.sinc(X_task_3 * 8 - 6).sum(axis=1)
        X_task_3 = np.concatenate((X_task_3, 2 * np.ones([10, 1])), axis=1)

        self.X = np.concatenate((X_task_1, X_task_2, X_task_3), axis=0)
        self.y = np.concatenate((y_task_1, y_task_2, y_task_3), axis=0)
        self.model = WrapperBohamiann()
        self.model.train(self.X, self.y)

    def test_predict(self):
        X_test = np.random.rand(10, 2)
        X_test = np.concatenate((X_test, np.ones([10, 1])), axis=1)

        m, v = self.model.predict(X_test)

        assert len(m.shape) == 1
        assert m.shape[0] == X_test.shape[0]
        assert len(v.shape) == 1
        assert v.shape[0] == X_test.shape[0]

    def test_get_incumbent(self):
        inc, inc_val = self.model.get_incumbent()

        b = np.argmin(self.y)
        np.testing.assert_almost_equal(inc, self.X[b], decimal=5)


if __name__ == "__main__":
    unittest.main()
