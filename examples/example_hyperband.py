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
sys.path.extend([gh_root + 'bandits/'])

import numpy as np

import ConfigSpace as CS
from hpolib.benchmarks.synthetic_functions import Forrester

from robo.solver.hyperband_datasets_size import HyperBand_DataSubsets




f = Forrester()

opt = HyperBand_DataSubsets(f, 3, 5)


opt.run(8)

from IPython import embed
embed()
