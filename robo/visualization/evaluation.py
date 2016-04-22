'''
Created on April 7th, 2016

@author: Numair Mansur (numair.mansur@gmail.com)
'''

import seaborn as sns
import matplotlib.pyplot as plt

def bar_plot(x, curves, title="", width=0.10,
            colors=['b', 'g', 'r', 'c', 'm', 'y', 'k'],
            log_scale_y=False, log_scale_x=False, legend=True,
            x_title="X Label", y_title="Y Label"):
    
    '''
    Plots Mean and Standard Deviation with a Bar Graph

    Example
    -------    
        x = np.array([[1, 3, 4, 5], [1, 3, 4, 5], [1, 3, 4, 5], [1, 2, 3, 4]])
        curve1 = np.array([[1, 3, 5, 7], [0.2, 0.4, 0.7, 0.4]])
        curve2 = np.array([[3, 2, 6, 8], [0.1, 1, 0.7, 0.3]])
        curve3 = np.array([[2, 4, 6, 3], [0.4, 0.4, 0.1, 0.3]])
        curve4 = np.array([[4, 3, 2, 1], [0.3, 0.4, 0.1, 0.2]])
        curves = [curve1, curve2, curve3, curve4]

        plot = bar_plot(x, curves, legend = True)
        plot.show()

    Parameters
    -----------
    x : numpy array
        For each curve, contains the x-coordinates. Each entry
        corresponds to one curve.
    curves : list of numpy arrays
        A list of 2D numpy arrays of mean and standard deviation. First entry in the
        numpy array corresponds to the mean and the second entry corresponds to the
        error. Each entry in the curves list corresponds to one curve.
    title : string
        Title of the graph
    width : float
    	Width of the bars.
    colors : string array
        Color of the curve. Each entry corresponds to one curve
    log_scale_y : Boolean
        If set to true, changes the y-axis to log scale.
    log_scale_x: Boolean
        If set to true, change the x-axis to log scale.
    legend : Boolean
        If set to true, displays the legend.
    x_title : String
    	X label string 
    y_title : String
    	Y label string

   	Retrun
   	----------
   	plt : object

    '''
    # Set Appearance properties using Seaborn
    sns.set(style="white", color_codes=True, font_scale=1.2)
    sns.set_style("ticks", {"xtick.major.size": 3, "ytick.major.size": 2})
    #sns.set_titles(col_template = "{col_name}", fontweight = 'bold', size = 18)
    # - - - - - - - -- - - - - - - - -
    x = x.astype(float)
    fig, ax = plt.subplots()
    bars = []
    # Stores the information about how many bars are saved on a X-Location.
    x_location_map = dict()

    # Pre-Process and adjust X-axis location data
    for i, j in enumerate(x):
        for l, k in enumerate(j):
            if k in x_location_map:
                x[i][l] = x[i][l] + (x_location_map[k] * width)
                x_location_map[k] += 1
            else:
                x_location_map[k] = 1
    # - - - - - - - - - - - - -
    for i, j in enumerate(x):
        bar = ax.bar(
            j,
            curves[i][0],
            width,
            yerr=curves[i][1],
            color=colors[i],
            error_kw=dict(
                ecolor='#525252',
                capsize=3,
                capthick=1.5))
        bars.append(bar)

    ax.set_ylabel(y_title, fontsize=14)
    ax.set_xlabel(x_title, fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.set_xticks([i + ((x_location_map[i] * width) / 2)
                   for i in x_location_map])
    ax.set_xticklabels([int(i) for i in x_location_map])
    if legend:
    	ax.legend(bars, ["Curve " + str(i + 1) for i, j in enumerate(bars)], loc=0)
    if log_scale_x:
        plt.xscale('log')
    if log_scale_y:
        plt.yscale('log')
    return plt


def plot_mean_and_std(x, curves, title="",
        colors=['b', 'g', 'r', 'c', 'm', 'y', 'k'],
        log_scale_y=False, log_scale_x=False, legend=True,
        x_title="X Label", y_title="Y Label", std_scale=1):

    '''
    Plots Mean and Standard Deviation with an error bar graph

    Example
    -------    

        x = np.array([[1, 3, 4, 5], [1, 3, 4, 5], [1, 3, 4, 5], [1, 2, 3, 4]])
        curve1 = np.array([[1, 3, 5, 7], [0.2, 0.4, 0.7, 0.4]])
        curve2 = np.array([[3, 2, 6, 8], [0.1, 1, 0.7, 0.3]])
        curve3 = np.array([[2, 4, 6, 3], [0.4, 0.4, 0.1, 0.3]])
        curve4 = np.array([[4, 3, 2, 1], [0.3, 0.4, 0.1, 0.2]])
        curves = [curve1, curve2, curve3, curve4]

        plot = point_plot(x, curves)
        plot.show()

    Parameters
    ----------
    x : numpy array
        For each curve, contains the x-coordinates. Each entry
        corresponds to one curve.
    curves : list of numpy arrays
        A list of 2D numpy arrays of mean and standard deviation. First entry in the
        numpy array corresponds to the mean and the second entry corresponds to the
        error. Each entry in the curves list corresponds to one curve.
    title : string
        Title of the graph
    colors : string array
        Color of the curve. Each entry corresponds to one curve
    log_scale_y : Boolean
        If set to true, changes the y-axis to log scale.
    log_scale_x: Boolean
        If set to true, change the x-axis to log scale.
    legend : Boolean
        If set to true, displays the legend.
    x_title : String
        X label string 
    y_title : String
        Y label string

    Retrun
    ----------
    plt : object

    '''
    # Set Appearance properties using Seaborn
    sns.set(style="white", color_codes=True, font_scale=1.2)
    sns.set_style("ticks", {"xtick.major.size": 3, "ytick.major.size": 2})
    # - - - - - - - - - - - - - - - - - - -
    plt.figure()

    for i, j in enumerate(x):
        plt.errorbar(j,
                     curves[i][0],
                     yerr=curves[i][1],
                     fmt='-',
                     marker="o",
                     label='curve' + str(i + 1),
                     color=colors[i])
    
    plt.ylabel(y_title, fontsize = 14)
    plt.xlabel(x_title, fontsize=13)    
    plt.title(title)
    plt.grid()
    if log_scale_x:
        plt.xscale('log')
    if log_scale_y:
        plt.yscale('log')
    if legend:
        plt.legend(loc=0)

    # Adjust Margins
    plot_margin = 0.25
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin,
              x1 + plot_margin,
              y0 - plot_margin,
              y1 + plot_margin))
    return plt


def latex_matrix_string(mean, error, title,
        row_labels, col_labels, best_bold_row=True, best_bold_column=False):
    
    '''
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
    '''
    matrix_string = '''\hline
'''
    for i, row in enumerate(mean):
        column_string = '''{ |c'''
        matrix_string = matrix_string + \
            "\\textbf{" + row_labels[i] + "}& "  # length of row labels and number of rows must be equal
        for j, cell in enumerate(row):
            column_string = column_string + '''|c'''
            ending_string = ''' & ''' if j < len(row) - 1 else ''' \\\ \hline
'''
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

