'''
Created on 2016/09/07
@author: Stefan Falkner
'''

import sys
gh_root= '/ihome/sfalkner/repositories/github/'
sys.path.extend([gh_root + 'RoBO/', gh_root + 'HPOlibConfigSpace/'])
sys.path.extend([gh_root + 'HPOlib/package/'])
sys.path.extend([gh_root + 'HPOlib/'])
bb_root= '/ihome/sfalkner/repositories/bitbucket/'
sys.path.extend([bb_root + 'bandits/'])

import numpy as np

import ConfigSpace as CS
import hpolib.benchmarks.synthetic_functions as hpobench
from robo.solver.hyperband_datasets_size import HyperBand_DataSubsets




f  = hpobench.SyntheticNoiseAndCost(hpobench.Forrester(), 0, 0.1, 2, 0, 1, 1)
opt = HyperBand_DataSubsets(f, 2, 0.05)


opt.run(5)

from IPython import embed
embed()
