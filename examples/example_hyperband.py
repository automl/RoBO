import hpolib.benchmarks.synthetic_functions as hpobench

from robo.solver.hyperband_datasets_size import HyperBand_DataSubsets


eta = 3.
B = 7

f = hpobench.SyntheticNoiseAndCost(hpobench.Forrester(), 0, 0.1, 1, 0, 1, 1)
opt = HyperBand_DataSubsets(f, eta, eta**(-(B-1)))


opt.run(8)

print(opt.time_func_eval)
