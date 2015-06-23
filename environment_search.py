'''
Created on Jun 23, 2015

@author: Aaron Klein
'''

import GPy
import errno
import argparse
import numpy as np

from robo.models import GPyModel
from robo.solver.env_bayesian_optimization import EnvBayesianOptimization
from robo.acquisition.EnvEntropySearch import EnvEntropySearch
from robo.maximizers import maximize

from ParameterConfigSpace.config_space import ConfigSpace


def parse_pcs_file(pcs_file):
    # Parse pcs file
    config_space = ConfigSpace(pcs_file)
    names = config_space.get_parameter_names()
    is_env = []
    dims = len(names)
    X_lower = np.zeros([dims])
    X_upper = np.zeros([dims])
    for i, name in enumerate(names):
        # Check if parameter is an environment variable
        is_env.append(name.starts_with("env_"))

        # Extract bounds
        X_lower[i] = config_space.parameters[name].values[0]
        X_upper[i] = config_space.parameters[name].values[1]
    return is_env, X_lower, X_upper, dims


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('R|'):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description='Run environment search',
                                     prog='environment_search',
                                     formatter_class=SmartFormatter)
    parser.add_argument('save_dir', metavar="DESTINATION_FOLDER", type=str)
    parser.add_argument('pcs_file', metavar="DESTINATION_FOLDER", type=str)
    parser.add_argument('--overwrite', action="store_true", default=False)
    parser.add_argument('-o', '--objective', default=None, type=str,
                        help='Choose your objective function',
                        required=True,
                        dest="objective")
    parser.add_argument('-p', '--acquisition-parameter', default=[], type=str, nargs="*",
                        help='''R|Choose the acquisition function parameters.
                        [float] For EI, PI, LogEI it\'s the minimum improvement.
                        [float] For UCB it\'s the factor the standard deviation is added.
                        [int] [int] For Entropy it's the number of representer points and the number of the predictions at an evaluation point 
                        [int] [int] [int] For EntropyMC it's the number of representer points, the number of the predictions at an evaluation point
                          followed by the number of functions drawn from the gp. ''',
                        dest="acquisition_parameters")
    parser.add_argument('-m', '--model', default="GPy", type=str,
                        choices=("GPy",),
                        help='Choose the model',
                        dest="model")
    parser.add_argument('-e', '--maximizer', default="stochastic_local_search", type=str,
                        choices=("grid_search", "DIRECT", "cma", "stochastic_local_search"),
                        help='Choose the acquisition maximizer',
                        dest="maximizer")
    parser.add_argument('-n', '--num', default="10", type=int,
                        help='number of function evaluations',
                        dest="n")
    parser.add_argument('-n_representer', '--num_representer', default="100", type=int,
                        help='Number of representer points',
                        dest="n_representer")
    parser.add_argument('-n_hals_vals', '--num_hals_vals', default="100", type=int,
                        help='Number of hallucinated values',
                        dest="n_hals_vals")
    parser.add_argument('-n_func_samples', '--num_func_samples', default="100", type=int,
                        help='Number of function samples',
                        dest="n_func_samples")
    parser.add_argument('--kernel_model', default="RBF", nargs="+", type=str,
                        help='Choose a kernel for GP based models',
                        choices=("RBF", "Linear", "Matern32", "Matern52", "Poly"),
                        dest="kernel_model")
    parser.add_argument('--kernel_cost_model', default="RBF", nargs="+", type=str,
                        help='Choose a kernel for GP based models',
                        choices=("RBF", "Linear", "Matern32", "Matern52", "Poly"),
                        dest="kernel_cost_model")
    parser.add_argument('--seed', default=None, type=str,
                        help='set a random seed',
                        dest="seed")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.overwrite == False:
        try:
            bo = EnvBayesianOptimization(save_dir=args.save_dir)
            bo.run(args.n, overwrite=False)
            exit()
        except OSError as exception:
            if exception.errno != errno.ENOENT:
                raise

    # TODO: Adjust the objective function
    args.objective

    is_env, X_lower, X_upper, dims = parse_pcs_file(args.pcs_file)

    # Specify the model and the cost model
    if args.model in ["GPy"]:
        if args.kernel == "RBF":
            kernel = GPy.kern.RBF(input_dim=dims)
        elif args.kernel == "Linear":
            kernel = GPy.kern.Linear(input_dim=dims)
        elif args.kernel == "Matern32":
            kernel = GPy.kern.Matern32(input_dim=dims)
        elif args.kernel == "Matern52":
            kernel = GPy.kern.Matern52(input_dim=dims)
        elif args.kernel == "Poly":
            kernel = GPy.kern.RBF(input_dim=dims)
    if args.cost_model in ["GPy"]:
        if args.kernel_cost_model == "RBF":
            cost_kernel = GPy.kern.RBF(input_dim=dims)
        elif args.kernel_cost_model == "Linear":
            cost_kernel = GPy.kern.Linear(input_dim=dims)
        elif args.kernel_cost_model == "Matern32":
            cost_kernel = GPy.kern.Matern32(input_dim=dims)
        elif args.kernel_cost_model == "Matern52":
            cost_kernel = GPy.kern.Matern52(input_dim=dims)
        elif args.kernel_cost_model == "Poly":
            cost_kernel = GPy.kern.RBF(input_dim=dims)

    model_kwargs = {}
    model_kwargs["noise_variance"] = args.noise_variance
    print args.noise_variance
    if args.model == "GPy":
        #TODO: MCMC sampling of the hyperparameters
        model = GPyModel(kernel, optimize=True, **model_kwargs)
        cost_model = GPyModel(cost_kernel, optimize=True, **model_kwargs)

    # Specify the acquisition function
    acquisition_fkt = EnvEntropySearch(model, cost_model,
                                       X_lower, X_upper,
                                       is_env,
                                       args.n_representer,
                                       args.n_hals_vals,
                                       args.n_func_samples,
                                       compute_incumbent)

    if args.maximizer == "grid_search":
        maximize_fkt = maximize.grid_search
    elif args.maximizer == "DIRECT":
        maximize_fkt = maximize.DIRECT
    elif args.maximizer == "cma":
        maximize_fkt = maximize.cma
    elif args.maximizer == "stochastic_local_search":
        maximize_fkt = maximize.stochastic_local_search

    #FIXME: Maybe it's a good idea to implement here the BO loop
    # Start the Bayesian optimization procedure
    bo = EnvBayesianOptimization(acquisition_fkt=acquisition_fkt,
                                 model=model,
                                 maximize_fkt=maximize_fkt,
                                 X_lower=X_lower,
                                 X_upper=X_upper,
                                 dims=dims,
                                 objective_fkt=objective_fkt,
                                 save_dir=args.save_dir,
                                 num_save=30)
    bo.run(args.n, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
