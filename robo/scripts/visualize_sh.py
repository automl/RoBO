import argparse
import os
import errno
from robo.visualization import Visualization
import robo
import numpy as np

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description='Visualize a robo run', 
                                     prog='robo_visualize')
                                     #formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source_folder', metavar="SOURCE_FOLDER", type=str)#, required=True)
    parser.add_argument('dest_folder',   metavar="DESTINATION_FOLDER", type=str)#, required=True)
    parser.add_argument('-a', '--acq',  default=False, 
                        help='Choose visualizing the acquisition fkt if its one_dimensional', 
                        dest="acq_method", action='store_true')
    parser.add_argument('-o', '--obj',  default=False, 
                        help='Choose visualizing the objective fkt if its one_dimensional', 
                        dest="obj_method", action='store_true')
    parser.add_argument('-m', '--model',  default=False, 
                        help='Choose visualizing the model fkt if its one_dimensional', 
                        dest="model_method", action='store_true')
    args = parser.parse_args()
    source_folder = args.source_folder
    dest_folder = args.dest_folder
    del args.source_folder
    del args.dest_folder
    i = 1
    if dest_folder is not None:
        try:
            os.makedirs(dest_folder)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    while True:
        try:
            bo, new_x, X, Y = robo.BayesianOptimization.from_iteration(source_folder, i)
            if bo.model_untrained:
                first_vis = args.__dict__.copy()
                del first_vis["acq_method"]
                del first_vis["model_method"]
                vis = Visualization(bo, new_x, X, Y, dest_folder, prefix="%03d_"%(i,), **first_vis)
            else: 
                print X, Y
                print i,": ", bo.model.m
                vis = Visualization(bo, new_x, X, Y, dest_folder, prefix="%03d_"%(i,), **args.__dict__)
            
            
            #del vis, bo, new_x, X, Y
            
        except IOError, e:
            break
        i += 1
    
        