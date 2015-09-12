# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:25:15 2015

@author: aaron
"""
import numpy as np


def test_performance(task, results):
    """
        Computes the test performance of a single run of an experiment. Returns a numpy array with the test performance of the incumbent after each
        iteration.
    """

    n_iters = len(results['incumbent'])
    test_performance = np.zeros([n_iters])
    for i in range(n_iters):
        test_performance[i] = task.evaluate_test(results['incumbent'][i][np.newaxis, :])[0]

    return test_performance


def distance_to_optimum(task, results):
    """

    """

    n_iters = len(results['incumbent'])
    dist = np.zeros([n_iters])
    for i in range(n_iters):
        dist[i] = np.linalg.norm(task.opt - results['incumbent'][i])

    return dist
