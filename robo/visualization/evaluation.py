import matplotlib.pyplot as plt
import numpy as np


def plot_over_iterations(x, methods, metric="mean", labels=None, linewidth=3,
                         x_label="Error", y_label="Number of iterations", log_y=False, log_x=False,
                         title="", legend_loc=1, percentiles=(5, 95), colors=None):
    """
    Plots performance over iterations of different methods .

    Example:
    ----------------------------
    x = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4]])
    method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
    method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
    method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
    methods = [method_1, method_2, method_3]
    plot = plot_median(x,methods)
    plot.show()

    Parameters:
    ----------
    x : numpy array
        For each curve, contains the x-coordinates. Each entry
        corresponds to one method.
    methods : list of numpy arrays
        A list of numpy arrays of methods. Each method contains a numpy array
        of several run of that corresponding method.
    method_names: List of Strings
        A list of names for the methods

    Returns
    -------
    plt : object
        Plot Object
    """

    if labels is None:
        labels = ["Method-%d" % i for i in range(len(methods))]

    styles = ["o", "D", "s", ">", "<", "^", "v", "*", "*", "."]

    if colors is None:
        colors = ["blue", "green", "purple", "darkorange", "red", "palevioletred", "lightseagreen", "brown", "black"]

    for index, method in enumerate(methods):
        style = styles[index % len(styles)]
        color = colors[index % len(colors)]
        if metric == "median":
            plt.plot(x[index], np.median(method, axis=0), label=labels[index], linewidth=linewidth, marker=style, color=color)
        elif metric == "mean":
            plt.plot(x[index], np.mean(method, axis=0), label=labels[index], linewidth=linewidth, marker=style, color=color)
        elif metric == "median_percentiles":
            plt.plot(x[index], np.median(method, axis=0), label=labels[index], linewidth=linewidth, marker=style, color=color)
            plt.fill_between(x[index], np.percentile(method, percentiles[0], axis=0),
                             np.percentile(method, percentiles[1], axis=0),
                             color=color, alpha=0.2)
        elif metric == "mean_std":
            plt.errorbar(x[index], np.mean(method, axis=0), yerr=np.std(method, axis=0),
                         label=labels[index], linewidth=linewidth, marker=style, color=color)
        else:
            raise ValueError("Metric does not exist!")

        plt.legend(loc=legend_loc)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if log_y:
            plt.yscale("log")
        if log_x:
            plt.xscale("log")
        plt.grid(True)
        plt.title(title)
    return plt


def plot_over_time(times, methods, metric="mean", labels=None, linewidth=3,
                         x_label="Error", y_label="Time", log_y=False, log_x=False,
                         title="", legend_loc=1, percentiles=(5, 95), colors=None, std_scale=1):
    """
    Plots performance over iterations

    Example:
    ----------------------------
    x = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4]])
    method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
    method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
    method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
    methods = [method_1, method_2, method_3]
    plot = plot_median(x,methods)
    plot.show()

    Parameters:
    ----------
    x : numpy array
        For each curve, contains the x-coordinates. Each entry
        corresponds to one method.
    methods : list of numpy arrays
        A list of numpy arrays of methods. Each method contains a numpy array
        of several run of that corresponding method.
    method_names: List of Strings
        A list of names for the methods

    Returns
    -------
    plt : object
        Plot Object
    """

    if labels is None:
        labels = ["Method-%d" % i for i in range(len(methods))]

    styles = ["o", "D", "s", ">", "<", "^", "v", "*", "*", "."]

    if colors is None:
        colors = ["blue", "green", "purple", "darkorange", "red", "palevioletred", "lightseagreen", "brown", "black"]

    for index, method in enumerate(methods):
        style = styles[index % len(styles)]
        color = colors[index % len(colors)]
        if metric == "median":
            plt.plot(times[index], np.median(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color)
        elif metric == "mean":
            plt.plot(times[index], np.mean(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color)
        elif metric == "median_percentiles":
            plt.plot(times[index], np.median(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color)
            plt.fill_between(times[index], np.percentile(method, percentiles[0], axis=0),
                             np.percentile(method, percentiles[1], axis=0),
                             color=color, alpha=0.2)
        elif metric == "mean_std":
            plt.step(times[index], np.mean(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color)
            plt.fill_between(times[index], np.mean(method, axis=0) + std_scale * np.std(method, axis=0),
                             np.mean(method, axis=0) - std_scale * np.std(method, axis=0),
                             linewidth=linewidth, color=color, alpha=0.2)
        else:
            raise ValueError("Metric does not exist!")

        plt.legend(loc=legend_loc)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if log_y:
            plt.yscale("log")
        if log_x:
            plt.xscale("log")
        plt.grid(True)
        plt.title(title)
    return plt
