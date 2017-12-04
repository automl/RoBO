import os
import sys
import json
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from robo.fmin import bayesian_optimization, entropy_search, bohamiann, random_search

from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM
from hpolib.benchmarks.ml.surrogate_cnn import SurrogateCNN
from hpolib.benchmarks.ml.surrogate_fcnet import SurrogateFCNet
from hpolib.benchmarks.ml.surrogate_paramnet import SurrogateParamNet


run_id = int(sys.argv[1])
benchmark = sys.argv[2]
method = sys.argv[3]

n_iters = 50
n_init = 2
output_path = "./experiments/RoBO/surrogates"

if benchmark == "svm_mnist":
    f = SurrogateSVM(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "cnn_cifar10":
    f = SurrogateCNN(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "fcnet_mnist":
    f = SurrogateFCNet(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "paramnet":
    dataset = sys.argv[4]
    f = SurrogateParamNet(dataset, "/ihome/kleinaa/devel/git/HPOlib/surrogates/")

    benchmark += "_" + dataset

info = f.get_meta_information()
bounds = np.array(info['bounds'])

if method == "entropy_search":
    results = entropy_search(f, bounds[:, 0], bounds[:, 1],
                             num_iterations=n_iters, n_init=n_init)
elif method == "gp_mcmc":
    results = bayesian_optimization(f, bounds[:, 0], bounds[:, 1],
                                    num_iterations=n_iters,
                                    n_init=n_init, model_type="gp_mcmc")
elif method == "gp":
    results = bayesian_optimization(f, bounds[:, 0], bounds[:, 1],
                                    num_iterations=n_iters,
                                    n_init=n_init, model_type="gp")
elif method == "rf":
    results = bayesian_optimization(f, bounds[:, 0], bounds[:, 1],
                                    num_iterations=n_iters,
                                    n_init=n_init, model_type="rf")
elif method == "random_search":
    results = random_search(f, bounds[:, 0], bounds[:, 1],
                            num_iterations=n_iters)
elif method == "bohamiann":
    results = bohamiann(f, bounds[:, 0], bounds[:, 1],
                        num_iterations=n_iters,
                        n_init=n_init)

# Offline Evaluation
test_error = []
cum_cost = 0

for i, inc in enumerate(results["incumbents"]):

    y = f.objective_function_test(np.array(inc))["function_value"]
    test_error.append(y)

    # Compute the time it would have taken to evaluate this configuration
    c = f.objective_function(np.array(results["X"][i]))["cost"]
    cum_cost += c

    # Estimate the runtime as the optimization overhead + estimated cost
    results["runtime"][i] += cum_cost
    results["test_error"] = test_error

results["method"] = method
results["benchmark"] = benchmark
results["run_id"] = run_id

p = os.path.join(output_path, benchmark, method)
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)
