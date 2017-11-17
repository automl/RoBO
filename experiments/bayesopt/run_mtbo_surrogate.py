import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.fmin import mtbo

from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM

run_id = int(sys.argv[1])
seed = int(sys.argv[2])
auxillay_dataset = int(sys.argv[3])
dataset = "surrogate"

rng = np.random.RandomState(seed)

f = SurrogateSVM(path="/home/kleinaa/experiments/fabolas/dataset/svm_on_mnist_grid", rng=rng)
num_iterations = 80
output_path = "./experiments/fabolas_journal/results/svm_%s/mtbo_%d_%d" % (dataset, auxillay_dataset, run_id)

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
               n_init=5, num_iterations=num_iterations, n_hypers=50,
               rng=rng, output_path=output_path, inc_estimation="last_seen")

results["run_id"] = run_id
results['X'] = results['X'].tolist()
results['y'] = results['y'].tolist()
results['c'] = results['c'].tolist()

test_error = []
current_inc = None
current_inc_val = None
cum_cost = 0

for i, inc in enumerate(results["incumbents"]):
    print(inc)
    if current_inc == inc:
        test_error.append(current_inc_val)
    else:
        y = f.objective_function_test(inc)["function_value"]
        test_error.append(y)

        current_inc = inc
        current_inc_val = y
    print(current_inc_val)

    # Compute the time it would have taken to evaluate this configuration
    c = results["c"][i]
    cum_cost += c

    # Estimate the runtime as the optimization overhead + estimated cost
    results["runtime"][i] += cum_cost
    results["test_error"] = test_error

    with open(os.path.join(output_path, 'results_%d.json' % run_id), 'w') as fh:
        json.dump(results, fh)
