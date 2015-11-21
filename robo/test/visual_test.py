'''
import GPy
import matplotlib; matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from robo.models.gpy_model import GPyModel
from robo.acquisition.ei import EI
from robo.maximizers.maximize import stochastic_local_search
from robo.recommendation.incumbent import compute_incumbent
from robo.visualization import plotting as plotting



# The optimization function that we want to optimize. It gets a numpy array with shape (N,D) where N >= 1 are the number of datapoints and D are the number of features
def objective_function(x):
    return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)

def run():
    # Defining the bounds and dimensions of the input space
    X_lower = np.array([0])
    X_upper = np.array([6])
    dims = 1

    # Set the method that we will use to optimize the acquisition function
    maximizer = stochastic_local_search

    # Defining the method to model the objective function
    kernel = GPy.kern.Matern52(input_dim=dims)
    model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)

    # The acquisition function that we optimize in order to pick a new x
    acquisition_func = EI(model, X_upper=X_upper, X_lower=X_lower, compute_incumbent=compute_incumbent, par=0.1)  # par is the minimum improvement that a point has to obtain

    # Draw one random point and evaluate it to initialize BO
    X = np.array([np.random.uniform(X_lower, X_upper, dims)])
    Y = objective_function(X)

    # Fit the model on the data we observed so far
    model.train(X, Y)
    # Update the acquisition function model with the retrained model
    acquisition_func.update(model)

    # Optimize the acquisition function to obtain a new point
    new_x = maximizer(acquisition_func, X_lower, X_upper)

    # Evaluate the point and add the new observation to our set of previous seen points
    new_y = objective_function(np.array(new_x))
    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)

    # Visualize the objective function, model and the acquisition function
    fig = plt.figure()
    #Sub plot for the model and the objective function
    ax1 = fig.add_subplot(2,1,1)
    #Sub plot for the acquisition function
    ax2 = fig.add_subplot(2,1,2)
    resolution = 0.1
    # Call plot_model function
    ax1=plotting.plot_model(model,X_lower,X_upper,ax1,resolution,'b','blue',"Prosterior Mean",3,True)
    #Call plot_objective_function
    ax1=plotting.plot_objective_function(objective_function,X_lower,X_upper,X,Y,ax1,resolution,'black','ObjectiveFunction',True)
    ax1.set_title("Model + Objective Function")
    #Call plot_acquisition_function
    ax2=plotting.plot_acquisition_function(acquisition_func,X_lower,X_upper,X,ax2,resolution,"AcquisitionFunction",True)
    plt.savefig('test2.png')
    os.system('eog test2.png&')
''' 