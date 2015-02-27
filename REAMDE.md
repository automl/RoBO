RoBO - a Robust Bayesian Optimization framework.
================================================

Installation
------------

	   

```
git clone git@bitbucket.org:aadfreiburg/robo.git
cd robo
pip install .
```

Some optional dependencies are the python packages `DIRECT` and `cma`


Example 
-------


```
#!python
#
import numpy as np
import GPy
from robo import BayesianOptimization
from robo.models import GPyModel 
from robo.acquisition import Entropy, LogEI, PI, EI
from robo.maximize import grid_search
from robo.loss_functions import logLoss
from robo.visualization import Visualization

#
# Defining our object of interest
#
X_lower = np.array([0])
X_upper = np.array([6])
dims = X_lower.shape[0]
def objective_funktion(x):
    return np.exp(x) / ((x+0.5)**2.0 * (np.sin(4.0*x)  + np.exp(1.0/3.0*x))) + np.random.normal(0, 0.01, x.shape)

num_initial = 4
initial_X = np.empty((num_initial, 1))
initial_X[0, :] = np.array([0.2])
initial_X[1:num_initial, :] = np.random.rand(num_initial-1, dims) * (X_upper - X_lower) + X_lower;
initial_Y = objective_funktion(initial_X)
print initial_X, initial_Y

#
# Creating our Robo Environment
#

kernel = GPy.kern.RBF(input_dim=dims)
maximize_fkt = grid_search
model = GPyModel(kernel, optimize=True, noise_variance = 1e-4, num_restarts=10)
entropy = Entropy(model, X_upper= X_upper, X_lower=X_lower, sampling_acquisition= LogEI, Nb=50, T=200, loss_function = logLoss)
ei = EI(model, X_upper= X_upper, X_lower=X_lower, par =0.3)
pi = PI(model, X_upper= X_upper, X_lower=X_lower, par =0.3)

for acquisition_fkt in [ei, pi, entropy]:
    
    bo = BayesianOptimization(acquisition_fkt=acquisition_fkt, 
                              model=model, 
                              maximize_fkt=maximize_fkt, 
                              X_lower=X_lower, 
                              X_upper=X_upper, 
                              dims=dims, 
                              objective_fkt=objective_funktion, 
                              save_dir=None)
    
    next_x = bo.choose_next(initial_X, initial_Y)
    
    Visualization(bo, 
                  next_x, 
                  X=initial_X, 
                  Y=initial_Y, 
                  dest_folder=None, 
                  show_acq_method = True, 
                  show_obj_method = True, 
                  show_model_method = True, 
                  resolution=1000, 
                  interactive=True)

```
	
This will produce an output similar to this

![Example output using EI](https://bitbucket.org/aadfreiburg/robo/readme_example_ei.png)
![Example output using PI](https://bitbucket.org/aadfreiburg/robo/readme_example_pi.png)
![Example output using Entropy](https://bitbucket.org/aadfreiburg/robo/readme_example_entropy.png)



An overview of RoBo
-------------------

Use `robo.BayesianOptimization(acquisition_fkt, model, maximize_fkt, X_lower, X_upper, dims, objective_fkt, save_dir)` with a combination of the following components:

+ **Acquisition Functions**: At the moment following acquisition functions are implemented
  - `EI` (Expected Improvement): It supports only one dimensional inputs and returns derivatives
  - `LogEI` (Log of Expected Improvement): It supports only one dimensional inputs
  - `PI` (Probability of Improvement): It supports only one dimensional inputs and returns derivatives
  - `UCB` (Upper Confidence Bound): Not well tested
  - `Entropy` (EP based Information Gain of the probability distribution of the Minimum): Works similar to the matlab code 
  - `EntropyMC` (Sample based Information Gain of the probability distribution of the Minimum): Work in progress
 
+ **Models**: 
  - `GPyModel`: A wrapper to the GPy GPRegression Class

+ **maximizers**: Maximizers of the acquisition functions
  - `cma`: works only for 2D+ objective functions
  - `DIRECT`: works for all dimensions. Needs an fortran compiler
  - `grid_search`: a trivial optimizer for 1D
 