'''
Created on April 7th, 2016
@author: Numair Mansur (numair.mansur@gmail.com)
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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
    	ax.legend(bars, ["Method " + str(i + 1) for i, j in enumerate(bars)], loc=0)
    if log_scale_x:
        plt.xscale('log')
    if log_scale_y:
        plt.yscale('log')
    return plt


def point_plot(x, curves, title="",
        colors=['b', 'g', 'r', 'c', 'm', 'y', 'k'],
        log_scale_y=False, log_scale_x=False, legend=True,
        x_title="X Label", y_title="Y Label"):

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
                     label='Method' + str(i + 1),
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



def plot_mean_and_std(x,methods,drawBarPlot = False, drawPointPlot = False, title="", width=0.10,
    colors=['b', 'g', 'r', 'c', 'm', 'y', 'k'], log_scale_y=False, log_scale_x=False, legend=True,
    x_title="X Label", y_title="Y Label"):
    '''
    Plots Mean and Standard Deviation of Methods with multiple runs

    Example
    -------    
x = np.array([[1, 3, 4, 5], [1, 3, 4, 5], [1, 3, 4, 6]])
method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
methods = [method_1, method_2, method_3]

plot = plot_mean_and_std(x,methods,drawBarPlot = True)
plot.show()

    Parameters
    -----------
    x : numpy array
        For each curve, contains the x-coordinates. Each entry
        corresponds to one method.
    methods : list of numpy arrays
        A list of numpy arrays of methods. Each method contains a numpy array
        of several run of that corresponding method.
    drawBarPlot : Bool
        Should be True if a Bar Plot is expected.
    drawPointPlot : Bool
        Should be True if a Point Plot is expected.
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
        Plot Object
    '''
    curves = []
    for index,method in enumerate(methods):
        mean = []
        std = []
        for j in range(0,len(x[index])):
            valueArray = np.array([el[j] for el in method])
            meanValue = np.mean(valueArray)
            stdValue = np.std(valueArray)
            mean.append(meanValue)
            std.append(stdValue)
        curves.append(np.array([mean,std]))
    if(drawBarPlot):
        barPlot = bar_plot(x,curves, title=title, width=width,
            colors=colors,
            log_scale_y=log_scale_y, log_scale_x=log_scale_x, legend=legend,
            x_title=x_title, y_title=y_title)
        return barPlot
    elif (drawPointPlot):
        pointPlot = point_plot(x,curves,title=title,
            colors=colors,
            log_scale_y=log_scale_y, log_scale_x=log_scale_x, legend=legend,
            x_title=x_title, y_title=y_title)
        return pointPlot
    else:
        raise NameError('Please select the type of the plot')


def plotStandardErrorOfMean(x,methods,drawBarPlot = False, drawPointPlot = False, title="", width=0.10,
    colors=['b', 'g', 'r', 'c', 'm', 'y', 'k'], log_scale_y=False, log_scale_x=False, legend=True,
    x_title="X Label", y_title="Y Label"):
    '''
    Plots Mean and Standard Error of the mean for Methods with multiple runs

    Example
    -------    
x = np.array([[1, 3, 4, 5], [1, 3, 4, 5], [1, 3, 4, 6]])
method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
methods = [method_1, method_2, method_3]

plot = plotStandardErrorOfMean(x,methods,drawBarPlot = True)
plot.show()

    Parameters
    -----------
    x : numpy array
        For each curve, contains the x-coordinates. Each entry
        corresponds to one method.
    methods : list of numpy arrays
        A list of numpy arrays of methods. Each method contains a numpy array
        of several run of that corresponding method.
    drawBarPlot : Bool
        Should be True if a Bar Plot is expected.
    drawPointPlot : Bool
        Should be True if a Point Plot is expected.
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
        Plot Object
    '''
    curves = []
    for index,method in enumerate(methods):
        mean = []
        sem = []
        for j in range(0,len(x[index])):
            valueArray = np.array([el[j] for el in method])
            meanValue = np.mean(valueArray)
            semValue = stats.sem(valueArray) #Standard Error of Mean
            mean.append(meanValue)
            sem.append(semValue)
        curves.append(np.array([mean,sem]))
    if(drawBarPlot):
        barPlot = bar_plot(x,curves, title=title, width=width,
            colors=colors,
            log_scale_y=log_scale_y, log_scale_x=log_scale_x, legend=legend,
            x_title=x_title, y_title=y_title)
        return barPlot
    elif (drawPointPlot):
        pointPlot = point_plot(x,curves,title=title,
            colors=colors,
            log_scale_y=log_scale_y, log_scale_x=log_scale_x, legend=legend,
            x_title=x_title, y_title=y_title)
        return pointPlot
    else:
        raise NameError('Please select the type of the plot')

x = np.array([[1, 3, 4, 5], [1, 3, 4, 5], [1, 3, 4, 6]])
method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
methods = [method_1, method_2, method_3]

plot = plotStandardErrorOfMean(x,methods,drawPointPlot = True)
plot.show()
def time_interpolation(time_point_union,data):
    '''
    Interpolates data over time points.
    Edits the original data dictionary. All of the methods should have runs on the same time point.
    i.e for each time point in the time_points_union array there should be a y value.
    :return:

    '''
    for method_number in time_point_union:
        for time_point in time_point_union[method_number]:
            for run_number in data[method_number]:
                if time_point in data[method_number][run_number]:
                    pass #Do nothing
                else: #Did not find the key
                    y_value_of_new_point = sorted([i for i in data[method_number][run_number] if i < time_point])
                    if not y_value_of_new_point:
                        y_value_of_new_point = sorted([i for i in data[method_number][run_number]])[-1]
                    else:
                        y_value_of_new_point = y_value_of_new_point[-1]
                    data[method_number][run_number][time_point] = {}
                    data[method_number][run_number][time_point] = data[method_number][run_number][y_value_of_new_point]
    return data



def plot_over_time(time,methods,error_random_config,agglomeration="mean"):
    """
    Takes different runs of a method with different time points and interpolates each run so that all the runs have a value
    at all the time points.
    Example:
    ---------
method1 = np.array([[80,84,85,82,83, 87,86,86,79,75,74],[53,52,59,54,55,56,54,59,54,52,50],[30,33,32,31,29, 28,26,27,26,24,23]])
method2 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
time = np.array([[[1, 2, 10, 15,16, 19,22,27,33,38,40], [1, 3, 9, 12,14, 19,21,30,35,40,42], [1, 3, 4, 6, 8, 20,22,28,33,45,46]],
        [[2,3,4,5], [4,5,6,7], [6,7,8,9]]])
methods = [method1,method2]
plot = plot_over_time(time,methods,0.9)
plot.show()
    Parameters:
    -----------


    :return:

    """
    # Transform data into a dictionary for more efficent and easy pre-processing.
    data = dict()
    time_point_union = dict()
    # Initializing the dictionary with data already available
    for i,method in enumerate(methods):
        data[i] = {}
        time_point_union[i] = list() # A dictionary of lists.
        for j,runs in enumerate(method):
            data[i][j] =dict()
            time_point_union[i] = np.union1d(time_point_union[i],time[i][j])
            time_point_union[i] = map(int,time_point_union[i])
            for k,y_value in enumerate(runs):
                data[i][j][time[i][j][k]]={}
                data[i][j][time[i][j][k]]= methods[i][j][k]
    data = time_interpolation(time_point_union,data) #Interpolating data on time points.
    # Transforming this dictionary into array format so that it can be visualised.
    new_time = list()
    new_methods = list()
    method_numbers = sorted(data.keys()) # A sorted list of method numbers.
    for i in method_numbers:
        new_time.append([time_point_union[i] for j in data[i]])
    new_time = np.asarray(new_time)
    # Use the dictionary to generate the new and interpolated methods array.
    for i in method_numbers:
        method1 = list()
        run_numbers = sorted(data[i].keys())
        for runs in run_numbers:
            array = [data[i][runs][time_point_union[i][o]] for o in [n for n in range(0,len(time_point_union[i]))]]
            method1.append(array)
        new_methods.append(method1)
    # Update the old data to the new data.
    time = new_time
    methods = new_methods
    for i,j in enumerate(methods):
        for l,m in enumerate(methods[i]):
            plt.step(time[i][l],m,where='post') #,color='r')
    return plt




def plot_median_and_percentiles(x,method, first_percentile = 5, second_percentile = 95):
    '''
    Plots the median and the percentiles of different runs of a method for a given time point.
    By default plots the 5th and 95th percentile, if values not given.
Example:
-----------
x = np.array([[10, 50, 70, 100 ], [10, 50, 70, 100], [10, 50, 70, 100]])
method_1 = np.array([[100,70,90,80], [110,70,100,70] , [90,70,80,85]])
method_2 = np.array([[50,58,65,45], [60,48,65,45] , [40,68,65,55]])
method_3 = np.array([[9,13,12,11], [11,13,9,11] , [12,13,5,11]])
methods = [method_1, method_2, method_3]
plot = plot_median_and_percentiles(x,methods)
plot.show()

:return:
    '''
    sns.set(style="white", color_codes=True, font_scale=1.2)
    color_above =['#e9967a', '#fa8072', '#ffa07a', '#ff8c00', '#ff7f50', '#f08080', '#ff6347','#ff4500','#ff0000']
    color_below = ['#6495ed', '#6a5acd', '#7b68ee', '#8470ff', '#4169e1', '#1e90ff', '#00bfff', '#87ceeb','#87cefa','#b0c4de','#add8e6','#b0e0e6','#afeeee','#00ced1']
    for index,method in enumerate(method):
        median = []
        fifth_percentile = []
        ninty_fifth_percentile = []
        for j in range(0,len(x[index])):
            valueArray = np.array([el[j] for el in method])
            medianValue = np.median(valueArray)
            fifth_value = np.percentile(valueArray, first_percentile)
            ninty_fifth_value = np.percentile(valueArray, second_percentile)
            ninty_fifth_percentile.append(ninty_fifth_value)
            fifth_percentile.append(fifth_value)
            median.append(medianValue)
        median_curve =np.array(median)
        ninty_fifth_percentile_curve = np.array(ninty_fifth_percentile)
        fifth_percentile_curve = np.array(fifth_percentile)
        plt.plot(x[index],median_curve, label = "median")
        plt.plot(x[index],ninty_fifth_percentile_curve, color = color_below[index])
        plt.plot(x[index],fifth_percentile_curve, color = color_above[index])
        plt.fill_between(x[index],median_curve,ninty_fifth_percentile_curve,color = color_below[index])
        plt.fill_between(x[index],median_curve,fifth_percentile_curve,color=color_above[index])
    return plt



def plot_median(x,methods, method_names=[]):
    '''
    Plots the median [ and the percentile value given? ]
    Example:
    ----------------------------
x = np.array([[1, 3, 4, 5], [1, 3, 4, 5], [1, 3, 4, 6]])
method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
methods = [method_1, method_2, method_3]
plot = plot_median(x,methods)
plot.show()
    Parameters:
    ----------------------------
        x : numpy array
            For each curve, contains the x-coordinates. Each entry
            corresponds to one method.
        methods : list of numpy arrays
            A list of numpy arrays of methods. Each method contains a numpy array
            of several run of that corresponding method.
    :return:
        plt : object
            Plot Object
    '''
    curves = []
    for index,method in enumerate(methods):
        median = []
        fifth_percentile = []
        ninty_fifth_percentile = []
        for j in range(0,len(x[index])):
            valueArray = np.array([el[j] for el in method])
            medianValue = np.median(valueArray)
            fifth_value = np.percentile(valueArray, 5)
            ninty_fifth_value = np.percentile(valueArray, 95)
            ninty_fifth_percentile.append(ninty_fifth_value)
            fifth_percentile.append(fifth_value)
            median.append(medianValue)
        curves.append(np.array(median))
        print ("5th-->" ,fifth_percentile)
        print ("95th --> " , ninty_fifth_percentile)
        print ("median --> " , median)
    for index,curve in enumerate(curves):
        plt.plot(x[index],curve, label=method_names[index]) if len(method_names) !=0 else plt.plot(x[index],curve)
        plt.legend()
    return plt



def plot_mean(x,methods,method_names=[]):
    '''
    Plots the mean of different runs of a mehtod for a given time point.
    Example:
    ----------------------------
x = np.array([[1, 3, 4, 5], [1, 3, 4, 5], [1, 3, 4, 6]])
method_1 = np.array([[1,4,5,2], [3,4,3,6] , [2,5,5,8]])
method_2 = np.array([[8,7,5,9], [7,3,9,1] , [3,2,9,4]])
method_3 = np.array([[10,13,9,11], [9,12,10,10] , [11,14,18,6]])
methods = [method_1, method_2, method_3]
method_names = ['test method 1','test method 2','test method 3']
plot = plot_mean(x,methods,method_names)
plot.show()

    Parameters:
    ----------------------------
        x : numpy array
            For each curve, contains the x-coordinates. Each entry
            corresponds to one method.
        methods : list of numpy arrays
            A list of numpy arrays of methods. Each method contains a numpy array
            of several run of that corresponding method.

    :return:
        plt : object
            Plot Object
    '''
    curves = []
    for index,method in enumerate(methods):
        mean = []
        for j in range(0,len(x[index])):
            valueArray = np.array([el[j] for el in method])
            meanValue = np.mean(valueArray)
            mean.append(meanValue)
        curves.append(np.array(mean))
    for index,curve in enumerate(curves):
        plt.plot(x[index],curve,label=method_names[index]) if len(method_names)!=0 else plt.plot(x[index],curve)
        plt.legend()
    return plt




'''
another heading called visualizations in the contents.
functionality
examples
containing plots.
'''