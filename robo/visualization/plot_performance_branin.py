import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np

from robo.visualization.trajectories import get_mean_and_var_performance_over_iterations
from robo.visualization.trajectories import get_mean_and_var_optimization_overhead_over_iterations


def plot_spearmint(ax, fmin):
    spearmint_results = np.load("/mhome/kleinaa/experiments/robo/branin/spearmint_trajectory.npy")
    mean = np.mean(spearmint_results, axis=0)
    mean = np.log(np.abs(mean - 0.397887))
    #std = np.std(spearmint_results, axis=0)
    iterations = np.arange(0, mean.shape[0], 1)
    ax.plot(iterations, mean, "red", label="spearmint")
    #ax.fill_between(iterations, mean + std, mean - std, facecolor="red", alpha=0.2)
    return ax


def main(names, dirs):
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    n_curves = len(names)
    color = ["blue", "green", "yellow", "black", "purple", "orange", "cyan", "magenta", "brown"]

    fmin = 0.397887
    for i in range(n_curves):
        paths = glob.glob(os.path.join(dirs[i], "run_*"))

        iterations, mean, var = get_mean_and_var_performance_over_iterations(paths)
        mean = np.log(np.abs(mean - fmin))
        #ax1.errorbar(iterations, mean, np.sqrt(var), fmt=color[i % len(color)], label=names[i])
        ax1.plot(iterations, mean, color[i % len(color)], label=names[i])
        #ax1.fill_between(iterations, mean + np.sqrt(var), mean - np.sqrt(var), facecolor=color[i % len(color)], alpha=0.2)

        iterations, mean, var = get_mean_and_var_optimization_overhead_over_iterations(paths)
        ax2.plot(iterations, mean, color[i % len(color)], label=names[i])
        ax2.fill_between(iterations, mean + np.sqrt(var), mean - np.sqrt(var), facecolor=color[i % len(color)], alpha=0.2)
        #ax2.errorbar(iterations, mean, np.sqrt(var), fmt=color[i % len(color)], label=names[i])

    ax1 = plot_spearmint(ax1, fmin)

    ax1.set_title("Performance")
    ax2.set_title("Optimization Overhead")
    ax1.set_ylabel('|fmin - fmin*|')
    ax2.set_ylabel('Seconds')
    ax2.set_xlabel('Number of function evaluations')
    plt.legend()
    plt.savefig("performance_branin.png")

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
