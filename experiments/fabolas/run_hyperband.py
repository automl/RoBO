import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.solver.hyperband_datasets_size import HyperBand_DataSubsets

from hpolib.benchmarks.ml.svm_benchmark import SvmOnMnist, SvmOnVehicle, SvmOnCovertype, SvmOnAdult, SvmOnHiggs, SvmOnLetter
from hpolib.benchmarks.ml.residual_networks import ResidualNeuralNetworkOnCIFAR10
from hpolib.benchmarks.ml.conv_net import ConvolutionalNeuralNetworkOnCIFAR10, ConvolutionalNeuralNetworkOnSVHN


run_id = int(sys.argv[1])
dataset = sys.argv[2]
seed = int(sys.argv[3])

rng = np.random.RandomState(seed)

if dataset == "mnist":
    f = SvmOnMnist(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
elif dataset == "vehicle":
    f = SvmOnVehicle(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
elif dataset == "covertype":
    f = SvmOnCovertype(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
elif dataset == "higgs":
    f = SvmOnHiggs(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
elif dataset == "adult":
    f = SvmOnAdult(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
elif dataset == "letter":
    f = SvmOnLetter(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
elif dataset == "cifar10":
    f = ConvolutionalNeuralNetworkOnCIFAR10(rng=rng)
    output_path = "./experiments/fabolas/results/cnn_%s/hyperband_%d" % (dataset, run_id)
elif dataset == "svhn":
    f = ConvolutionalNeuralNetworkOnSVHN(rng=rng)
    output_path = "./experiments/fabolas/results/cnn_%s/hyperband_%d" % (dataset, run_id)
elif dataset == "res_net":
    f = ResidualNeuralNetworkOnCIFAR10(rng=rng)
    output_path = "./experiments/fabolas/results/res_%s/hyperband_%d" % (dataset, run_id)


os.makedirs(output_path, exist_ok=True)

eta = 3.
B = -int(np.log(f.s_min)/np.log(3))

print(B)

opt = HyperBand_DataSubsets(f, eta, eta**(-(B-1)), output_path=output_path, rng=rng)

opt.run(int(20 / B * 1.5))

test_error = []
for c in opt.incumbents:
    test_error.append(f.objective_function_test(c)["function_value"])

    results = dict()

    results["test_error"] = test_error
    results["runtime"] = opt.runtime
    results["time_func_eval"] = opt.time_func_eval_incumbent
    results["run_id"] = run_id

    with open(os.path.join(output_path, 'results_%d.json' % run_id), 'w') as fh:
        json.dump(results, fh)
