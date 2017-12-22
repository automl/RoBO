import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.fmin import mtbo

from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM
from hpolib.benchmarks.ml.surrogate_cnn import SurrogateCNN
from hpolib.benchmarks.ml.surrogate_fcnet import SurrogateFCNet


run_id = int(sys.argv[1])
benchmark = sys.argv[2]
auxillay_dataset = int(sys.argv[3])

rng = np.random.RandomState(run_id)

if benchmark == "svm_mnist":
    f = SurrogateSVM(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "cnn_cifar10":
    f = SurrogateCNN(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")
elif benchmark == "fcnet_mnist":
    f = SurrogateFCNet(path="/ihome/kleinaa/devel/git/HPOlib/surrogates/")

num_iterations = 80
output_path = "./experiments/RoBO/surrogate/"

os.makedirs(output_path, exist_ok=True)


def objective(x, task):
    if task == 0:
        dataset_fraction = float(1/auxillay_dataset)
    elif task == 1:
        dataset_fraction = 1

    res = f.objective_function(x, dataset_fraction=dataset_fraction)
    return res["function_value"], res["cost"]

info = f.get_meta_information()
bounds = np.array(info['bounds'])
results = mtbo(objective_function=objective,
               lower=bounds[:, 0], upper=bounds[:, 1],
               n_init=5, num_iterations=num_iterations,
               n_hypers=50,rng=rng)

results["run_id"] = run_id
results['X'] = results['X'].tolist()
results['y'] = results['y'].tolist()
results['c'] = results['c'].tolist()

test_error = []
current_inc = None
current_inc_val = None
cum_cost = 0

for i, inc in enumerate(results["incumbents"]):

    if current_inc == inc:
        test_error.append(current_inc_val)
    else:
        y = f.objective_function_test(inc)["function_value"]
        test_error.append(y)

        current_inc = inc
        current_inc_val = y

    # Compute the time it would have taken to evaluate this configuration
    c = results["c"][i]
    cum_cost += c

    # Estimate the runtime as the optimization overhead + estimated cost
    results["runtime"][i] += cum_cost
    results["test_error"] = test_error

p = os.path.join(output_path, benchmark, "mtbo_%d" % auxillay_dataset)
os.makedirs(p, exist_ok=True)

fh = open(os.path.join(p, '%s_run_%d.json' % (benchmark, run_id)), 'w')
json.dump(results, fh)
fh.close()
