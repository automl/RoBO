import unittest
import numpy as np

from robo.util.mc_part import joint_pmin


class TestMCPart(unittest.TestCase):

    def test_joint_pmin(self):
        m = np.zeros([2, 1])
        v = np.diag(np.ones(2))
        pmin = joint_pmin(m, v, 10000)

        np.testing.assert_allclose(pmin, np.array([0.5, 0.5]), 1e-1)

if __name__ == "__main__":
    unittest.main()
