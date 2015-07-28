'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

import os
import glob
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np


def get_performance_over_iterations(path):
    """
        Returns the performance of the incumbent after each iteration

    :param path: Path to the pickle files of this run
    """
    num_iters = len(glob.glob(os.path.join(path, "iteration_*")))

    iterations = np.arange(0, num_iters)
    performance = np.zeros([num_iters])

    for i in range(num_iters):
        file_name = "iteration_%03d.pkl" % (i)
        if not os.path.isfile(os.path.join(path, file_name)):
            print "ERROR: File $s does not exists!" % (file_name)
        if len( pickle.load(open(os.path.join(path, file_name), "r"))) == 7:
            _, _, _, _, incumbent_value, _, _ = pickle.load(open(os.path.join(path, file_name), "r"))
        else:
            _, _, _, incumbent_value, _, _ = pickle.load(open(os.path.join(path, file_name), "r"))

        if type(incumbent_value) == np.ndarray:
            performance[i] = incumbent_value[0]

    return iterations, performance


def get_mean_and_var_performance_over_iterations(paths):
    """
        Returns the mean and variance of the performances for different runs. Note: the number of iterations of each runs has to be the same.

    :param paths: A list of paths to the output of the different runs
    """

    performances = []
    iterations = None
    for path in paths:
        if not os.path.isdir(path):
            print "ERROR: %s is not a directory!" % (path)
            return
        iters, perf = get_performance_over_iterations(path)
        if (np.any(iterations is None)):
            iterations = iters
        else:
            assert iterations.shape == iters.shape
        performances.append(perf)

    performances = np.array(performances)

    return iterations, np.mean(performances, axis=0), np.var(performances, axis=0)


def get_performance_over_time(path):
    """
        Returns the performance of the incumbent over time

    :param path: Path to the pickle files of this run
    """
    num_iters = len(glob.glob(os.path.join(path, "iteration_*")))

    time = np.zeros([num_iters])
    performance = np.zeros([num_iters])

    for i in range(num_iters):
        file_name = "iteration_%03d.pkl" % (i)
        if not os.path.isfile(os.path.join(path, file_name)):
            print "ERROR: File $s does not exists!" % (file_name)
        if len( pickle.load(open(os.path.join(path, file_name), "r"))) == 7:
            _, _, _, _, incumbent_value, time_func_eval, _ = pickle.load(open(os.path.join(path, file_name), "r"))
        else:
            _, _, _, incumbent_value, time_func_eval, _ = pickle.load(open(os.path.join(path, file_name), "r"))

        performance[i] = incumbent_value[0]
        time[i] = time_func_eval

    return time, performance


def get_optimization_overhead_over_iteration(path):
    num_iters = len(glob.glob(os.path.join(path, "iteration_*")))

    iterations = np.arange(0, num_iters)
    optimization_overhead = np.zeros([num_iters])

    for i in range(num_iters):
        file_name = "iteration_%03d.pkl" % (i)
        if not os.path.isfile(os.path.join(path, file_name)):
            print "ERROR: File $s does not exists!" % (file_name)

        if len( pickle.load(open(os.path.join(path, file_name), "r"))) == 7:
            _, _, _, _, _, _, time_optimization_overhead = pickle.load(open(os.path.join(path, file_name), "r"))
        else:
            _, _, _, _, _, time_optimization_overhead = pickle.load(open(os.path.join(path, file_name), "r"))

        optimization_overhead[i] = time_optimization_overhead[0]

    return iterations, optimization_overhead


def get_mean_and_var_optimization_overhead_over_iterations(paths):
    """
        Returns the mean and variance of the optimization overhead for different runs. Note: the number of iterations of each runs has to be the same.

    :param paths: A list of paths to the output of the different runs
    """

    optimization_overhead = []
    iterations = None
    for path in paths:
        if not os.path.isdir(path):
            print "ERROR: %s is not a directory!" % (path)
            return
        iters, t = get_optimization_overhead_over_iteration(path)
        if (np.any(iterations is None)):
            iterations = iters
        else:
            assert iterations.shape == iters.shape
        optimization_overhead.append(t)

    optimization_overhead = np.array(optimization_overhead)

    return iterations, np.mean(optimization_overhead, axis=0), np.var(optimization_overhead, axis=0)


def evaluate_test_performance(task, trajectory, save_name=None):
    trajectory_test_perf = np.zeros([trajectory.shape[0]])
    for idx, config in enumerate(trajectory):
        trajectory_test_perf[idx] = task.evaluate_test(config)
    if save_name is not None:
        np.save(save_name, trajectory_test_perf)
    return trajectory_test_perf

