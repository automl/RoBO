import unittest
import numpy as np

from robo.models.bayesian_neural_network import BayesianNeuralNetwork
import tensorflow as tf


class TestBayesianNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)

        graph = tf.Graph()
        with graph.as_default():
            self.session = tf.Session(graph=graph)

            self.model = BayesianNeuralNetwork(
                session=self.session, dtype=tf.float64,
                normalize_output=True, normalize_input=True,
                mdecay=0.05
            )
            self.model.train(self.X, self.y)

    def test_predict(self):
        with self.session.graph.as_default():
            X_test = np.random.rand(10, 2)

            m, v = self.model.predict(X_test)

            assert len(m.shape) == 1
            assert m.shape[0] == X_test.shape[0]
            assert len(v.shape) == 1
            assert v.shape[0] == X_test.shape[0]

    def test_get_incumbent(self):
        with self.session.graph.as_default():
            self.model.train(self.X, self.y)

            inc, inc_val = self.model.get_incumbent()

            b = np.argmin(self.y)
            np.testing.assert_almost_equal(inc, self.X[b], decimal=5)


if __name__ == "__main__":
    unittest.main()
