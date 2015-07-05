"""
Contextual Bayesian optimization, example
"""

import pickle

import matplotlib
import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt

import GPy
from robo.contextual_bayesian_optimization import ContextualBayesianOptimization
from robo.models.GPyModel import GPyModel
from robo.acquisition.UCB import UCB
#from robo.maximizers.maximize import stochastic_local_search as maximizer
from robo.maximizers.maximize import cma as maximizer
#from robo.maximizers.maximize import grid_search as maximizer

from example_contextual_objective1 import objective1
from example_contextual_objective1 import objective1_min
from example_contextual_objective1 import objective1_S_lower
from example_contextual_objective1 import objective1_S_upper
from example_contextual_objective1 import objective1_dims_Z
from example_contextual_objective1 import objective1_dims_S


from example_contextual_objective2 import objective2
from example_contextual_objective2 import objective2_min
from example_contextual_objective2 import objective2_S_lower
from example_contextual_objective2 import objective2_S_upper
from example_contextual_objective2 import objective2_dims_Z
from example_contextual_objective2 import objective2_dims_S

# Kernel combinations
kernelpairs = [(GPy.kern.Matern32, GPy.kern.Matern32, "Matern 3/2", "Matern 3/2"),
               (GPy.kern.Matern52, GPy.kern.Matern52, "Matern 5/2", "Matern 5/2"),
               (GPy.kern.RBF, GPy.kern.RBF, "RBF", "RBF"),
               (GPy.kern.Matern32, GPy.kern.Matern52, "Matern 3/2", "Matern 5/2"),
               (GPy.kern.Matern52, GPy.kern.Matern32, "Matern 5/2", "Matern 3/2"),
               (GPy.kern.Matern32, GPy.kern.RBF, "Matern 3/2", "RBF"),
               (GPy.kern.RBF, GPy.kern.Matern32, "RBF", "Matern 3/2"),
               (GPy.kern.Matern52, GPy.kern.RBF, "Matern 5/2", "RBF"),
               (GPy.kern.RBF, GPy.kern.Matern52, "RBF", "Matern 5/2"),]

# Context function acquires random values
def context_fkt():
    return np.random.uniform(size=(1,2))

objectives = [(objective1, context_fkt, 'bran', 'product of two Branin function', objective1_min, objective1_S_lower, objective1_S_upper, objective1_dims_Z, objective1_dims_S),
              (objective2, context_fkt, 'hart', 'Hartmann 6 function', objective2_min, objective2_S_lower, objective2_S_upper, objective2_dims_Z, objective2_dims_S)]

num_iterations = 32
num_retries = 4

data = []

for objective, context_fkt, name_short, obj_name, objective_min, S_lower, S_upper, dims_Z, dims_S in objectives:
    f1, axes1 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(17.5, 10))
    axes1 = [ax for ax in axes1[0]] + [ax for ax in axes1[1]] + [ax for ax in axes1[2]]

    f2, axes2 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(17.5, 10))
    axes2 = [ax for ax in axes2[0]] + [ax for ax in axes2[1]] + [ax for ax in axes2[2]]

    kernels = [(opsign, opname, kernel1name, kernel2name, kernel1, kernel2, kernelop, axes[kernelid])
               for kernelid, (kernel1, kernel2, kernel1name, kernel2name) in enumerate(kernelpairs)
               for kernelop, opsign, opname, axes in [
                   (lambda kernel1, kernel2, dims_Z, dims_X: kernel1(input_dim=dims_Z, active_dims=list(range(dims_Z))) *
                                                             kernel2(input_dim=dims_S, active_dims=list(range(dims_Z, dims_X))),
                    '$\\cdot$', 'mul', axes1),
                   (lambda kernel1, kernel2, dims_Z, dims_X: kernel1(input_dim=dims_Z, active_dims=list(range(dims_Z))) +
                                                             kernel2(input_dim=dims_S, active_dims=list(range(dims_Z, dims_X))),
                    '+', 'add', axes2)]]

    for opsign, opname, kernel1name, kernel2name, kernel1, kernel2, kernelop, ax in kernels:
        for _ in range(num_retries):
            kernel = kernelop(kernel1, kernel2, dims_Z, dims_Z + dims_S)
            X_lower = np.concatenate((np.tile(np.array([-np.inf]), (dims_Z,)), S_lower))
            X_upper = np.concatenate((np.tile(np.array([np.inf]), (dims_Z,)), S_upper))

            # Defining the method to model the objective function
            model = GPyModel(kernel, optimize=True, noise_variance=1e-2, num_restarts=10, optimize_args={'optimizer': 'tnc'})

            # The acquisition function that we optimize in order to pick a new x
            acquisition_func = UCB(model, X_lower=X_lower, X_upper=X_upper, par=1.0)

            bo = ContextualBayesianOptimization(acquisition_fkt=acquisition_func,
                                                model=model,
                                                maximize_fkt=maximizer,
                                                S_lower=S_lower,
                                                S_upper=S_upper,
                                                dims_Z=dims_Z,
                                                dims_S=dims_S,
                                                objective_fkt=objective,
                                                context_fkt=context_fkt)

            print "Result:", bo.run(num_iterations=num_iterations)

            # Calculate regret
            real_data = objective_min(Z=bo.X[:, :bo.dims_S]).flatten()
            pred_data = bo.Y.flatten()
            regret = np.maximum(pred_data - real_data, 0)
            cum_regret = np.cumsum(regret)
            contextual_regret = cum_regret / np.arange(1, len(cum_regret) + 1)

            # Save the data
            data.append(regret)
            data.append(contextual_regret)

            # Plot data
            ax.set_xlabel('iterations')
            ax.set_ylabel('regret')
            ax.set_title('Context kernel %s %s\nAction kernel %s' % (kernel1name, opsign, kernel2name))
            plt1, = ax.plot(regret, 'r^--', label='Regret')
            plt2, = ax.plot(contextual_regret, 'bo--', label='Contextual Regret')
            #ax.get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xticks(list(range(num_iterations)))
            #ax.legend(handles=(plt1, plt2))
        ax.legend(handles=(plt1, plt2))
        break
    #fig.suptitle('Regret of %s' % obj_name)
    f1.tight_layout()
    f1.savefig('%s_mul.svg' % name_short)
    f2.tight_layout()
    f2.savefig('%s_add.svg' % name_short)

with open('example_contextual.pkl', 'wb') as outputf:
    pickle.dump(data, outputf, -1)

plt.show(block=True)