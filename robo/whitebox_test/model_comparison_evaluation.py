# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:40:32 2015

@author: aaron
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from robo.util.output_reader import OutputReader
from robo.util.offline_test import test_performance
from robo.task.within_model_comparison import WithinModelComparison


entropy = np.zeros([20, 40])
entropy_mc = np.zeros([20, 40])
ei = np.zeros([20, 40])
pi = np.zeros([20, 40])
ucb = np.zeros([20, 40])

o = OutputReader()            

for i in range(1, 20):
        task = WithinModelComparison(seed=i+10)
        # Entropy MC
        output_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/entropy_mc/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        entropy_mc[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
        print task.fopt

        # EI
        output_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/ei/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        ei[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
    
        # PI
        output_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/pi/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        pi[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
    
        # UCB
        output_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/ucb/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        ucb[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
    
        # Entropy
        output_dir = "/home/aaron/experiment/model_comparison/" + str(i) + "/entropy/"
        output = o.read_results_file(os.path.join(output_dir, "results.csv"))
        entropy[i] = np.log(np.abs(task.fopt - test_performance(task, output)))
    
iters = np.arange(40)
plt.plot(iters, entropy_mc.mean(axis=0), color='blue', label='entropy_mc')
plt.plot(iters, entropy.mean(axis=0), color='green', label='entropy')
plt.plot(iters, ei.mean(axis=0), color='red', label='ei')
plt.plot(iters, pi.mean(axis=0), color='orange', label='pi')
plt.plot(iters, ucb.mean(axis=0), color='purple', label='ucb')
plt.legend()
plt.ylabel("log(|fstar - f|)")
plt.xlabel("# evals")
plt.show()