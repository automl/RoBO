import sys
sys.path.insert(0, '../')
import unittest
import numpy as np
# import GPy
from models import GPyModel
from acquisition import EI

class EITestCase1(unittest.TestCase):

    def setUp(self):
        self.x = np.array([-4.0623,-0.7667])


    def test(self):
        print self.x
        acq_fn = EI(self.x)
        ei_value = acq_fn(self.x)
        print type(ei_value)

if __name__=="__main__":
    unittest.main()
