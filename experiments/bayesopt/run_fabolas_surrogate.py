import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.fmin import fabolas

from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM
from hpolib.benchmarks.ml.surrogate_cnn import SurrogateCNN
from hpolib.benchmarks.ml.surrogate_fcnet import SurrogateFCNet


run_id = int(sys.argv[1])
benchmark = sys.argv[2]

if benchmark == "svm_mnist":
    f = SurrogateSVM(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "cnn_cifar10":
    f = SurrogateCNN(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "fcnet_mnist":
    f = SurrogateFCNet(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")

output_path = "./experiments/RoBO/surrogates"

rng = np.random.RandomState(run_id)

num_iterations = 150
s_max = 50000
s_min = 100
subsets = [128] * 8
subsets.extend([64] * 4)
subsets.extend([32] * 2)
subsets.extend([4] * 1)


def objective(x, s):
    dataset_fraction = s / s_max

    res = f.objective_function(x, dataset_fraction=dataset_fraction)
    return res["function_value"], res["cost"]

info = f.get_meta_information()
bounds = np.array(info['bounds'])
lower = bounds[:, 0]
upper = bounds[:, 1]
results = fabolas(objective_function=objective, lower=lower, upper=upper,
                  s_min=s_min, s_max=s_max, n_init=len(subsets), num_iterations=num_iterations,
                  n_hypers=30, subsets=subsets, inc_estimation="mean",
                  rng=rng)

results["run_id"] = run_id
results['X'] = results['X'].tolist()
results['y'] = results['y'].tolist()
results['c'] = results['c'].tolist()

test_error = []
cum_cost = 0

for i, inc in enumerate(results["incumbents"]):
    y = f.objective_function_test(np.array(inc))["function_value"]
    test_error.append(y)

    # Compute the time it would have taken to evaluate this configuration
    c = results["c"][i]
    cum_cost += c

    # Estimate the runtime as the optimization overhead + estimated cost
    results["runtime"][i] += cum_cost
    results["test_error"] = test_error

results["method"] = "fabolas"
results["benchmark"] = benchmark
results["run_id"] = run_id

p = os.path.join(output_path, benchmark, "fabolas")
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)
