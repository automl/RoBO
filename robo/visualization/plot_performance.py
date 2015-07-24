'''
Created on Jun 23, 2015

@author: Aaron Klein
'''
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np

from robo.visualization.trajectories import get_mean_and_var_performance_over_iterations
from robo.visualization.trajectories import get_mean_and_var_optimization_overhead_over_iterations


def main(names, dirs):
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    n_curves = len(names)
    color = ["blue", "green", "red", "yellow", "black", "purple", "orange", "cyan", "magenta", "brown"]

    for i in range(n_curves):
        paths = glob.glob(os.path.join(dirs[i], "run_*"))

        iterations, mean, var = get_mean_and_var_performance_over_iterations(paths)
        ax1.errorbar(iterations, mean, np.sqrt(var), fmt=color[i % len(color)], label=names[i])

        iterations, mean, var = get_mean_and_var_optimization_overhead_over_iterations(paths)
        ax2.errorbar(iterations, mean, np.sqrt(var), fmt=color[i % len(color)], label=names[i])

    ax1.set_title("Performance")
    ax2.set_title("Optimization Overhead")
    ax1.set_ylabel('fmin')
    ax2.set_ylabel('Seconds')
    ax2.set_xlabel('Number of function evaluations')

    plt.legend()
    plt.savefig("performance_over_iterations_branin.png")

if __name__ == '__main__':
    args = sys.argv[1:]
    names = []
    dirs = []
    for e in range(len(args)):
        if (e % 2) == 0:
            names.append(args[e])
        elif(e % 2) == 1:
            dirs.append(args[e])

    main(names, dirs)
