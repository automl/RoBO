import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.solver.hyperband_datasets_size import HyperBand_DataSubsets
from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM


run_id = int(sys.argv[1])
seed = int(sys.argv[2])

rng = np.random.RandomState(seed)
dataset = "surrogate"

f = SurrogateSVM(path="/mhome/kleinaa/experiments/fabolas/dataset/svm_on_mnist_grid", rng=rng)
output_path = "/mhome/kleinaa/experiments/fabolas_journal/results/svm_%s/hyperband_last_seen_incumbent_%d" % (dataset, run_id)

os.makedirs(output_path, exist_ok=True)

eta = 3.
B = -int(np.log(f.s_min)/np.log(3))

print(B)

opt = HyperBand_DataSubsets(f, eta, eta**(-(B-1)), output_path=output_path, rng=rng)

opt.run(int(20 / B * 1.5))

test_error = []
runtime = []
cum_cost = 0
for i, c in enumerate(opt.incumbents):
    test_error.append(f.objective_function_test(c)["function_value"])

    results = dict()

    results["test_error"] = test_error

    cum_cost += opt.time_func_eval_incumbent[i]
    runtime.append(opt.runtime[i] + cum_cost)
    results["runtime"] = runtime

    results["run_id"] = run_id

    with open(os.path.join(output_path, 'results_%d.json' % run_id), 'w') as fh:
        json.dump(results, fh)
