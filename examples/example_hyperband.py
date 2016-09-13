'''
Created on 2016/09/07
@author: Stefan Falkner
'''

import sys
gh_root= '/ihome/sfalkner/repositories/github/'
sys.path.extend([gh_root + 'RoBO/', gh_root + 'ConfigSpace/'])
sys.path.extend([gh_root + 'HPOlib/'])
bb_root= '/ihome/sfalkner/repositories/bitbucket/'
sys.path.extend([bb_root + 'bandits/'])

import numpy as np

import ConfigSpace as CS
import hpolib.benchmarks.synthetic_functions as hpobench
from robo.solver.hyperband_datasets_size import HyperBand_DataSubsets

eta = 3.
B=5

f  = hpobench.SyntheticNoiseAndCost(hpobench.Forrester(), 0, 0.1, 1, 0, 1, 1)
opt = HyperBand_DataSubsets(f, eta, eta**(-B))


opt.run(B)

print(opt.time_func_eval)


from IPython import embed
embed()
