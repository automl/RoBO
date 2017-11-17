import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from collections import OrderedDict


def latex_matrix_string(mean, title,
                        row_labels, col_labels,
                        best_bold_row=True, best_bold_column=False):
    """
    Latex Matrix String Generator.

    Example
    -------
    mean = [[1, 6, 5, 7], [12, 4, 6, 13], [9, 8, 7, 10]]
    print(latex_matrix_string(mean, "Testing Testing", [
                     "row1", "row2", "row3"], [
                     "col1", "col2", "col3", "col4"]))

    Parameters
    ----------
    mean : array of float array
            An array of float arrays containing mean values
    title : string
            Title string of the table
    row_labels : string array
            Array of strings for row names
    col_labels : string arrays
            Array of strings for column names
    best_bold_row : boolean
            If set to true, the minimum mean entry in each row will
            be set to bold.
    best_bold_column :
            If set to true, the minimum mean entry in each column will
            be set to bold.
    """
    matrix_string = '''\hline
'''
    for i, row in enumerate(mean):
        column_string = '''{ |c'''
        matrix_string = matrix_string + \
                        "\\textbf{" + row_labels[i] + "}& "  # length of row labels and number of rows must be equal
        for j, cell in enumerate(row):
            column_string = column_string + '''|c'''
            ending_string = ''' & ''' if j < len(row) - 1 else ''' \\\ \hline'''
            if best_bold_row and cell == min(
                    row) and best_bold_column == False:
                matrix_string = matrix_string + \
                                "$\mathbf{" + str(cell) + "}$" + ending_string
            elif best_bold_column and cell == min([a[j] for a in mean]) and best_bold_row == False:
                matrix_string = matrix_string + \
                                "$\mathbf{" + str(cell) + "}$" + ending_string
            else:
                matrix_string = matrix_string + "$" + \
                                str(cell) + "$" + ending_string
    column_string = column_string + '''| }'''
    column_label = ""
    for column in col_labels:
        column_label = column_label + "&\\textbf{" + column + "}"
    latex_string1 = '''\\begin{table}[ht]
\centering
\\begin{tabular}
''' + column_string + '''
\hline
''' + column_label + "\\\ [0.1ex]" + '''
''' + matrix_string + '''\end{tabular}
\\\[-1.5ex]
\caption{''' + title + '''}
\end{table}'''
    return latex_string1


def latex_matrix_string_mean_error(mean, error, title,
                                   row_labels, col_labels,
                                   best_bold_row=True, best_bold_column=False):
    """
    Latex Matrix String Generator.

    Example
    -------
    mean = [[1, 6, 5, 7], [12, 4, 6, 13], [9, 8, 7, 10]]
    error = [[2, 6, 1, 5], [4, 8, 2, 3], [1, 4, 8, 2]]
    print(latex_matrix_string(mean, error, "Testing Testing", [
                     "row1", "row2", "row3"], [
                     "col1", "col2", "col3", "col4"]))

    Parameters
    ----------
    mean : array of float array
            An array of float arrays containing mean values
    error : array of float array
            An array of float array containing error values
    title : string
            Title string of the table
    row_labels : string array
            Array of strings for row names
    col_labels : string arrays
            Array of strings for column names
    best_bold_row : boolean
            If set to true, the minimum mean entry in each row will
            be set to bold.
    best_bold_column :
            If set to true, the minimum mean entry in each column will
            be set to bold.
    """
    matrix_string = '''\hline
'''
    for i, row in enumerate(mean):
        column_string = '''{ |c'''
        matrix_string = matrix_string + \
                        "\\textbf{" + row_labels[i] + "}& "  # length of row labels and number of rows must be equal
        for j, cell in enumerate(row):
            column_string = column_string + '''|c'''
            ending_string = ''' & ''' if j < len(row) - 1 else ''' \\\ \hline'''
            if best_bold_row and cell == min(
                    row) and best_bold_column == False:
                matrix_string = matrix_string + \
                                "$\mathbf{" + str(cell) + " \pm " + str(error[i][j]) + "}$" + ending_string
            elif best_bold_column and cell == min([a[j] for a in mean]) and best_bold_row == False:
                matrix_string = matrix_string + \
                                "$\mathbf{" + str(cell) + " \pm " + str(error[i][j]) + "}$" + ending_string
            else:
                matrix_string = matrix_string + "$" + \
                                str(cell) + " \pm " + str(error[i][j]) + "$" + ending_string
    column_string = column_string + '''| }'''
    column_label = ""
    for column in col_labels:
        column_label = column_label + "&\\textbf{" + column + "}"
    latex_string1 = '''\\begin{table}[ht]
\centering
\\begin{tabular}
''' + column_string + '''
\hline
''' + column_label + "\\\ [0.1ex]" + '''
''' + matrix_string + '''\end{tabular}
\\\[-1.5ex]
\caption{''' + title + '''}
\end{table}'''
    return latex_string1


# def plot_over_iterations(x, methods, metric="mean", labels=None, linewidth=3, fontsize_label=25,
#                          x_label="Error", y_label="Number of iterations", log_y=False, log_x=False,
#                          title="", legend_loc=1, percentiles=(5, 95), colors=None, plot_legend=True):
#     """
#     Plots performance over iterations of different methods .
#
#     Example:
#     ----------------------------
#     x = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4]])
#     method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
#     method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
#     method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
#     methods = [method_1, method_2, method_3]
#     plot = plot_median(x,methods)
#     plot.show()
#
#     Parameters:
#     ----------
#     x : numpy array
#         For each curve, contains the x-coordinates. Each entry
#         corresponds to one method.
#     methods : list of numpy arrays
#         A list of numpy arrays of methods. Each method contains a numpy array
#         of several run of that corresponding method.
#     method_names: List of Strings
#         A list of names for the methods
#
#     Returns
#     -------
#     plt : object
#         Plot Object
#     """
#
#     if labels is None:
#         labels = ["Method-%d" % i for i in range(len(methods))]
#
#     styles = ["o", "D", "s", ">", "<", "^", "v", "*", "*", ".", ",", "1", "2", "3", "4"]
#
#     if colors is None:
#         colors = ["blue", "green", "purple", "darkorange", "red",
#                   "palevioletred", "lightseagreen", "brown", "black",
#                   "firebrick", "cyan", "gold", "slategray"]
#
#     for index, method in enumerate(methods):
#         style = styles[index % len(styles)]
#         color = colors[index % len(colors)]
#         if metric == "median":
#             plt.plot(x[index], np.median(method, axis=0), label=labels[index], linewidth=linewidth, marker=style,
#                      color=color)
#         elif metric == "mean":
#             plt.plot(x[index], np.mean(method, axis=0), label=labels[index], linewidth=linewidth, marker=style,
#                      color=color)
#         elif metric == "median_percentiles":
#             plt.plot(x[index], np.median(method, axis=0), label=labels[index], linewidth=linewidth, marker=style,
#                      color=color)
#             plt.fill_between(x[index], np.percentile(method, percentiles[0], axis=0),
#                              np.percentile(method, percentiles[1], axis=0),
#                              color=color, alpha=0.2)
#         elif metric == "mean_std":
#             plt.errorbar(x[index], np.mean(method, axis=0), yerr=np.std(method, axis=0),
#                          label=labels[index], linewidth=linewidth, marker=style, color=color)
#         else:
#             raise ValueError("Metric does not exist!")
#
#         if plot_legend:
#             plt.legend(loc=legend_loc, fancybox=True, framealpha=1, frameon=True, fontsize=fontsize_label)
#
#         plt.xlabel(x_label, fontsize=fontsize_label)
#         plt.ylabel(y_label, fontsize=fontsize_label)
#         plt.grid(True, which='both', ls="-")
#         if log_y:
#             plt.yscale("log")
#         if log_x:
#             plt.xscale("log")
#
#         plt.title(title, fontsize=fontsize_label)
#     return plt

def fill_trajectory(performance_list, time_list, replace_nan=np.NaN):
    frame_dict = OrderedDict()
    counter = np.arange(0, len(performance_list))
    for p, t, c in zip(performance_list, time_list, counter):
        if len(p) != len(t):
            raise ValueError("(%d) Array length mismatch: %d != %d" %
                             (c, len(p), len(t)))
        frame_dict[str(c)] = pd.Series(data=p, index=t)

    merged = pd.DataFrame(frame_dict)
    merged = merged.ffill()

    performance = merged.get_values()
    time_ = merged.index.values

    performance[np.isnan(performance)] = replace_nan

    if not np.isfinite(performance).all():
        raise ValueError("\nCould not merge lists, because \n"
                         "\t(a) one list is empty?\n"
                         "\t(b) the lists do not start with the same times and"
                         " replace_nan is not set?\n"
                         "\t(c) any other reason.")

    return performance, time_


def plot_optimization_trajectories(times, methods, metric="mean", labels=None, linewidth=3, fontsize=20,
                                   x_label="Error", y_label="Time", log_y=False, log_x=False, plot_legend=True,
                                   title="", legend_loc=0, percentiles=(5, 95), colors=None, std_scale=1,
                                   markersize=15, markevery=1):
    """
    Plot performance over time

    Example:
    ----------------------------
    x = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4]])
    method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
    method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
    method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
    methods = [method_1, method_2, method_3]
    plot = plot_optimization_trajectories(x, methods)
    plot.show()

    """

    if labels is None:
        labels = ["Method-%d" % i for i in range(len(methods))]

    styles = ["o", "D", "s", ">", "<", "^", "v", "*", "*", "."]

    if colors is None:
        colors = ["blue", "green", "purple", "darkorange",
                  "red", "palevioletred", "lightseagreen", "brown", "black"]

    for index, method in enumerate(methods):
        style = styles[index % len(styles)]
        color = colors[index % len(colors)]
        if metric == "median":
            plt.plot(times[index], np.median(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color, markevery=markevery,
                     markersize=markersize, markeredgecolor=color, markerfacecolor='none', markeredgewidth=5)
        elif metric == "mean":
            plt.plot(times[index], np.mean(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color, markevery=markevery,
                     markersize=markersize, markeredgecolor=color, markerfacecolor='none', markeredgewidth=5)
        elif metric == "median_percentiles":
            plt.plot(times[index], np.median(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color, markevery=markevery,
                     markersize=markersize, markeredgecolor=color, markerfacecolor='none', markeredgewidth=5)
            plt.fill_between(times[index], np.percentile(method, percentiles[0], axis=0),
                             np.percentile(method, percentiles[1], axis=0),
                             color=color, alpha=0.2)
        elif metric == "mean_std":
            plt.plot(times[index], np.mean(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color, markevery=markevery,
                     markersize=markersize, markeredgecolor=color, markerfacecolor='none', markeredgewidth=5)
            plt.fill_between(times[index], np.mean(method, axis=0) + std_scale * np.std(method, axis=0),
                             np.mean(method, axis=0) - std_scale * np.std(method, axis=0),
                             linewidth=linewidth, color=color, alpha=0.2)
        elif metric == "mean_sem":
            plt.plot(times[index], np.mean(method, axis=0), label=labels[index],
                     linewidth=linewidth, marker=style, color=color, markevery=markevery,
                     markersize=markersize, markeredgecolor=color, markerfacecolor='none', markeredgewidth=5)
            plt.fill_between(times[index], np.mean(method, axis=0) + std_scale * stats.sem(method, axis=0),
                             np.mean(method, axis=0) - std_scale * stats.sem(method, axis=0),
                             linewidth=linewidth, color=color, alpha=0.2)
        else:
            raise ValueError("Metric does not exist!")

        if plot_legend:
            plt.legend(loc=legend_loc, fancybox=True, framealpha=1, frameon=True, fontsize=fontsize)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        if log_y:
            plt.yscale("log")
        if log_x:
            plt.xscale("log")
        plt.grid(True, which='both', ls="-", alpha=1)
        plt.title(title, fontsize=fontsize)
    return plt
