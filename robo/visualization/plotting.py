
import numpy as np

try:
    import seaborn
    seaborn.set_style(style='whitegrid')
except:
    pass



def plot_model(model, X_lower, X_upper, ax,
        resolution=0.1, maximizer=None, mean_color='b',
        uncertainty_color='orange', label="Model", std_scale=1,
        plot_mean=True):            
    """
    Plots the mean and std of the model on a regular grid of input point

    Parameters
    ----------
    model: BaseModel
        The model that captures the posterior of the objective function
    X_lower: np.array
        Upper bound of the input space
    X_upper: np.array
        Lower bound of the input space
    ax: matplotlib figure
        Subplot for the model and the objective funciton
    resolution: float
        Resolution of the input points
    mean_color: string
        Specifies the color of the prosterior mean
    uncertainity_color: string
        Specifies the color of the model
    label: string
        Label that will appear in the legend
    std_scale: int
        Scales the standard deviation
    plot_mean: bool
        Bool flag, plots the mean curve if value is True

    Returns:
    --------        
    ax: figure
        subplot for the model and the objective funciton
    """

    X = np.arange(X_lower[0], X_upper[0] + resolution, resolution)

    mean = np.zeros([X.shape[0]])
    var = np.zeros([X.shape[0]])
    ax.grid()
    for i in range(X.shape[0]):
        mean[i], var[i] = model.predict(X[i, np.newaxis, np.newaxis])
    var[var < 0.0] = 0.0
    if plot_mean:
        ax.plot(X, mean, mean_color, label=label, linewidth=3)
    if maximizer is not None:
        ax.axvline(maximizer, color='red')
    ax.fill_between(X, mean + std_scale * np.sqrt(var),
        mean - std_scale * np.sqrt(var),
        facecolor=uncertainty_color,
        alpha=0.4)

    if label != None:
        ax.legend()
    return ax


def plot_objective_function(task, ax, X=None, Y=None,
        resolution=0.1, color='black', color_points='red',
        label='ObjectiveFunction'):
    """
    Plots the objective function on a regular grid of input point

    Parameters
    ----------
    task: BaseTask
        Task object that contains the objective function
    ax: matplotlib figure
        Subplot for the model and the objective funciton
    X: np.ndarray(N, D)
        The observed datapoints
    Y: np.ndarray(N, 1)
        The function values of the observed datapoints        
    resolution: float
        Resolution of the input points
    color: string
        Specifies the color of the objective function
    color_points: string
        Specifies the color of the observed points
    label: string
        Label that will appear in the legend


    Returns:
    --------        
    ax: figure
        subplot for the model and the objective funciton
    """
    grid = np.arange(task.X_lower[0], task.X_upper[0] + resolution, resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in range(grid.shape[0]):
        grid_values[i] = task.evaluate(np.array([[grid[i]]]))[0]

    ax.plot(grid, grid_values, color, label=label, linestyle="--")
    ax.grid()
    if X is not None and Y is not None:
        ax.scatter(X, Y, color=color_points)
    if label != None:
        ax.legend()
    return ax


def plot_acquisition_function(acquisition_function,
        X_lower, X_upper, ax, resolution=0.1, label="BaseAcquisitionFunction",
        maximizer=None):
    """
    Plots the acquisition function on a regular grid of input point

    Parameters
    ----------
    acquisition function: BaseAcquisition
        The acquisition function object
    X_lower: np.array
        Upper bound of the input space
    X_upper: np.array
        Lower bound of the input space
    ax: matplotlib figure
        Subplot for the model and the objective funciton
    resolution: float
        Resolution of the input points
    label: string
        Label that will appear in the legend
    maximizer: BaseMaximizer
        Maxmimizer object, if not none this object is called to find
        the max of the acquisition function

    Returns:
    --------        
    ax: figure
        subplot for the model and the objective funciton
    """
    grid = np.arange(X_lower[0], X_upper[0] + resolution, resolution)

    grid_values = np.zeros([grid.shape[0]])
    for i in range(grid.shape[0]):
        grid_values[i] = acquisition_function(grid[i, np.newaxis])
    #grid_values[grid_values < 0.0] = 0.0
    ax.plot(grid, grid_values, "g", label=label)
    ax.grid()
    if maximizer is not None:
        ax.axvline(maximizer, color='red')
    if label != None:
        ax.legend()
    return ax
