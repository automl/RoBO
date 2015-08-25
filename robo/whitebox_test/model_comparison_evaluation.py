# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:40:32 2015

@author: aaron
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from robo.util.output_reader import OutputReader
from robo.util.offline_test import test_performance, distance_to_optimum
from robo.task.within_model_comparison import WithinModelComparison


entropy = np.zeros([40, 100])
entropy_mc = np.zeros([15, 100])
ei = np.zeros([40, 100])
pi = np.zeros([40, 100])
ucb = np.zeros([40, 100])

dist_entropy = np.zeros([40, 100])
dist_entropy_mc = np.zeros([15, 100])
dist_ei = np.zeros([40, 100])
dist_pi = np.zeros([40, 100])
dist_ucb = np.zeros([40, 100])

o = OutputReader()

for i in range(1, 40):
        task = WithinModelComparison(seed=i + 10)
        # Entropy MC
        if i < 14:
            output_dir = "/mhome/kleinaa/experiments/entropy_search/model_comparison/func_" + str(i) + "/entropy_mc/"
            output = o.read_results_file(os.path.join(output_dir, "results.csv"))
            entropy_mc[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
            dist_entropy_mc[i] = np.log(distance_to_optimum(task, output))

        # EI
        output_dir = "/mhome/kleinaa/experiments/entropy_search/model_comparison/func_" + str(i) + "/ei/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        ei[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
        dist_ei[i] = np.log(distance_to_optimum(task, output))

        # PI
        output_dir = "/mhome/kleinaa/experiments/entropy_search/model_comparison/func_" + str(i) + "/pi/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        pi[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
        dist_pi[i] = np.log(distance_to_optimum(task, output))

        # UCB
        output_dir = "/mhome/kleinaa/experiments/entropy_search/model_comparison/func_" + str(i) + "/ucb/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        ucb[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
        dist_ucb[i] = np.log(distance_to_optimum(task, output))

        # Entropy
        output_dir = "/mhome/kleinaa/experiments/entropy_search/model_comparison/func_" + str(i) + "/entropy/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        entropy[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
        dist_entropy[i] = np.log(distance_to_optimum(task, output))

iters = np.arange(100)
plt.plot(iters, entropy_mc.mean(axis=0), color='blue', label='entropy_mc')
plt.plot(iters, entropy.mean(axis=0), color='green', label='entropy')
plt.plot(iters, ei.mean(axis=0), color='red', label='ei')
plt.plot(iters, pi.mean(axis=0), color='orange', label='pi')
plt.plot(iters, ucb.mean(axis=0), color='purple', label='ucb')
plt.legend()
plt.ylabel("log(|fstar - f|)")
plt.xlabel("# evals")
plt.show()

plt.plot(iters, dist_entropy_mc.mean(axis=0), color='blue', label='entropy_mc')
plt.plot(iters, dist_entropy.mean(axis=0), color='green', label='entropy')
plt.plot(iters, dist_ei.mean(axis=0), color='red', label='ei')
plt.plot(iters, dist_pi.mean(axis=0), color='orange', label='pi')
plt.plot(iters, dist_ucb.mean(axis=0), color='purple', label='ucb')
plt.legend()
plt.ylabel("log(|xstar - x|)")
plt.xlabel("# evals")
plt.show()
