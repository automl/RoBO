import os
import random
import argparse
import errno
import GPy
import pylab as pb
import numpy as np

from robo import BayesianOptimization
from robo.models import GPyModel
from robo.test_functions import one_dim_test, branin, hartmann6, hartmann3, goldstein_price_fkt
from robo.acquisition import EI, PI, LogEI, Entropy, UCB, EntropyMC
from robo.maximize import grid_search, DIRECT, cma, stochastic_local_search


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('R|'):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description='Run RoBO',
                                     prog='python run.py',
                                     formatter_class=SmartFormatter)

    parser.add_argument('save_dir', metavar="DESTINATION_FOLDER", type=str)

    parser.add_argument('--overwrite', action="store_true", default=False)

    parser.add_argument('-o', '--objective', default=None, type=str,
                        help='Choose your objective function',
                        required=True,
                        dest="objective")

    parser.add_argument('-a', '--acquisition', default=None, type=str,
                        help='Choose the acquisition function',
                        required=True,
                        dest="acquisition", choices=("EI", "PI", "LogEI", "Entropy", "EntropyMC", "UCB"))

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

    parser.add_argument('-e', '--maximizer', default="", type=str,
                        choices=("grid_search", "DIRECT", "cma", "stochastic_local_search"),
                        required=True,
                        help='Choose the acquisition maximizer',
                        dest="maximizer")

    parser.add_argument('-n', '--num', default="10", type=int,
                        help='number of evaluations',
                        dest="n")

    parser.add_argument('--noise', help='noise variance. Defaults None. If noise is None then it will be optimized.',
                        dest="noise_variance", default=None)

    parser.add_argument('--kernel', default="RBF", nargs="+", type=str,
                        help='Choose a kernel for GP based models',
                        choices=("RBF",),
                        dest="kernel")
    parser.add_argument('--seed', default=None, type=str,
                        help='set a random seed',
                        dest="seed")
    parser.add_argument('--pcs_file',
                        default=[],
                        type=str,
                        help='Parameter configuration space file (in SMAC format)',
                        required=True,
                        dest="pcs_file")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

#     if args.overwrite == False:
#         try:
#             bo = BayesianOptimization(save_dir=args.save_dir)
#             bo.run(args.n, overwrite=False)
#             exit()
#         except OSError as exception:
#             if exception.errno != errno.ENOENT:
#                 raise

    objective_fkt = args.objective

    dims = objective_fkt.dims
    X_lower = objective_fkt.X_lower
    X_upper = objective_fkt.X_upper

    #
    # Building up the model
    #
    if args.model in ("GPy",) and args.kernel == "RBF":
        kernel = GPy.kern.RBF(input_dim=dims)

    model_kwargs = {}

    model_kwargs["noise_variance"] = args.noise_variance
    print args.noise_variance
    if args.model == "GPy":
        model = GPyModel(kernel, optimize=True, **model_kwargs)

    #
    # creating an acquisition function
    #
    acquisition_kwargs = {}
    if args.acquisition == "EI":
        if len(args.acquisition_parameters):
            acquisition_kwargs["par"] = float(args.acquisition_parameters[0])
        acquisition_fkt = EI(model, X_upper=X_upper, X_lower=X_lower, **acquisition_kwargs)
    elif args.acquisition == "PI":
        if len(args.acquisition_parameters):
            acquisition_kwargs["par"] = float(args.acquisition_parameters[0])
        acquisition_fkt = PI(model, X_upper=X_upper, X_lower=X_lower, **acquisition_kwargs)
    elif args.acquisition == "LogEI":
        if len(args.acquisition_parameters):
            acquisition_kwargs["par"] = float(args.acquisition_parameters[0])
        acquisition_fkt = LogEI(model, X_upper=X_upper, X_lower=X_lower, **acquisition_kwargs)
    elif args.acquisition == "Entropy":
        if len(args.acquisition_parameters):
            acquisition_kwargs["Nb"] = int(args.acquisition_parameters[0])
            acquisition_kwargs["Np"] = int(args.acquisition_parameters[1])
        acquisition_fkt = Entropy(model, X_upper=X_upper, X_lower=X_lower, **acquisition_kwargs)
    elif args.acquisition == "EntropyMC":
        if len(args.acquisition_parameters):
            acquisition_kwargs["Nb"] = int(args.acquisition_parameters[0])
            acquisition_kwargs["Np"] = int(args.acquisition_parameters[1])
            acquisition_kwargs["Nf"] = int(args.acquisition_parameters[2])
        acquisition_fkt = EntropyMC(model, X_upper=X_upper, X_lower=X_lower, **acquisition_kwargs)
    elif args.acquisition == "UCB":
        if len(args.acquisition_parameters):
            acquisition_kwargs["par"] = float(args.acquisition_parameters[0])
        acquisition_fkt = UCB(model, X_upper=X_upper, X_lower=X_lower, **acquisition_kwargs)
    if args.maximizer == "grid_search":
        maximize_fkt = grid_search
    elif args.maximizer == "DIRECT":
        maximize_fkt = DIRECT
    elif args.maximizer == "cma":
        maximize_fkt = cma
    elif args.maximizer == "stochastic_local_search":
        maximize_fkt = stochastic_local_search

    bo = BayesianOptimization(acquisition_fkt=acquisition_fkt,
                              model=model,
                              maximize_fkt=maximize_fkt,
                              X_lower=X_lower,
                              X_upper=X_upper,
                              dims=dims,
                              objective_fkt=objective_fkt,
                              save_dir=args.save_dir,
                              num_save=30)

    X = np.random.randn(1, 2)
    print objective_fkt

    bo.run(args.n, overwrite=args.overwrite)

if __name__ == "__main__":
    main()