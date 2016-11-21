import george
import unittest
import numpy as np

from robo.acquisition_functions.information_gain import InformationGain
from robo.models.gaussian_process import GaussianProcess
from robo.util import epmgp


class TestInformationGain(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.y = np.sinc(self.X * 10 - 5).sum(axis=1)
        kernel = george.kernels.Matern52Kernel(np.array([0.1, 0.1]), ndim=2)
        self.model = GaussianProcess(kernel)
        self.model.train(self.X, self.y)
        self.acquisition_func = InformationGain(self.model, np.zeros([2]), np.ones([2]))
        self.acquisition_func.update(self.model)

    def test_compute(self):
        X_test = np.random.rand(5, 2)
        a = self.acquisition_func.compute(X_test, derivative=False)
        assert a.shape[0] == X_test.shape[0]
        assert len(a.shape) == 1

    def test_sampling_representer_points(self):

        # Check if representer points are inside the bounds
        assert np.any(self.acquisition_func.zb >= self.acquisition_func.lower)
        assert np.any(self.acquisition_func.zb <= self.acquisition_func.upper)

    def test_compute_pmin(self):

        # Uniform distribution
        m = np.ones([self.acquisition_func.Nb])
        v = np.eye(self.acquisition_func.Nb)

        pmin = epmgp.joint_min(m, v)
        pmin = np.exp(pmin)

        uprob = 1. / self.acquisition_func.Nb

        assert pmin.shape[0] == self.acquisition_func.Nb
        assert np.any(pmin < (uprob + 0.03)) and np.any(pmin > uprob - 0.01)

        # Dirac delta
        m = np.ones([self.acquisition_func.Nb]) * 1000
        m[0] = 1
        v = np.eye(self.acquisition_func.Nb)

        pmin = epmgp.joint_min(m, v)
        pmin = np.exp(pmin)
        uprob = 1. / self.acquisition_func.Nb
        assert pmin[0] == 1.0
        assert np.any(pmin[:1] > 1e-10)

    def test_innovations(self):
        # Case 1: Assume no influence of test point on representer points
        rep = np.array([[1.0]])
        x = np.array([[0.0]])
        dm, dv = self.acquisition_func.innovations(x, rep)

        assert np.any(np.abs(dm) < 1e-3)
        assert np.any(np.abs(dv) < 1e-3)

        # Case 2: Test point is close to representer points
        rep = np.array([[1.0]])
        x = np.array([[0.99]])
        dm, dv = self.acquisition_func.innovations(x, rep)
        assert np.any(np.abs(dm) > 1e-3)
        assert np.any(np.abs(dv) > 1e-3)

if __name__ == "__main__":
    unittest.main()
