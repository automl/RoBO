# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:25:52 2015

@author: aaron
"""


import unittest
import numpy as np

from robo.priors.default_priors import TophatPrior
from robo.priors.default_priors import LognormalPrior
from robo.priors.default_priors import HorseshoePrior


class TestTophatPrior(unittest.TestCase):

    def test(self):
        l = -2
        u = 2
        prior = TophatPrior(l, u)
        
        # Check sampling
        p0 = prior.sample_from_prior(10)
        assert len(p0.shape) == 2
        assert p0.shape[0] == 10
        assert p0.shape[1] == 1

        # Check gradients
        
        # Check likelihood
        theta = np.array([0])
        assert prior.lnprob(theta) == 0
        theta = np.array([-3])
        assert prior.lnprob(theta) == -np.inf


class TestHorseshoePrior(unittest.TestCase):

    def test(self):
        prior = HorseshoePrior()
        
        # Check sampling
        p0 = prior.sample_from_prior(10)
        assert len(p0.shape) == 2
        assert p0.shape[0] == 10
        assert p0.shape[1] == 1

        # Check gradients
 

class TestLognormalPrior(unittest.TestCase):

    def test(self):
        prior = LognormalPrior(sigma=0.1)
        
        # Check sampling
        p0 = prior.sample_from_prior(10)
        assert len(p0.shape) == 2
        assert p0.shape[0] == 10
        assert p0.shape[1] == 1

        # Check gradients
               

if __name__ == "__main__":
    unittest.main()
