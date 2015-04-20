import argparse
import os
import errno
import copy

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt;
from robo.util.visualization import Visualization
import robo
import numpy as np


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description='Visualize a robo run',
                                     prog='robo_visualize')
                                     # formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source_folder', metavar="SOURCE_FOLDER", type=str)  # , required=True)
    parser.add_argument('dest_folder', metavar="DESTINATION_FOLDER", type=str)  # , required=True)
    parser.add_argument('-a', '--acq', default=False,
                        help='Choose visualizing the acquisition fkt if its one_dimensional',
                        dest="show_acq_method", action='store_true')
    parser.add_argument('-o', '--obj', default=False,
                        help='Choose visualizing the objective fkt if its one_dimensional',
                        dest="show_obj_method", action='store_true')
    parser.add_argument('-m', '--model', default=False,
                        help='Choose visualizing the model fkt if its one_dimensional',
                        dest="show_model_method", action='store_true')
    parser.add_argument('--incumbent_gap', default=False,
                        help='Plot Incumbent over time',
                        dest="show_incumbent_gap", action='store_true')
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
            
    incumbent_list_x = None
    incumbent_list_y = None
    
    args_dict = args.__dict__.copy()
    del args_dict["show_incumbent_gap"]
    while True:
        try:
            bo, new_x, X, Y, buest_guess = robo.BayesianOptimization.from_iteration(source_folder, i)
            if args.show_incumbent_gap:
                if not bo.model_untrained:
                    if incumbent_list_x is None:
                        incumbent_list_x = np.empty((0, X.shape[1]), dtype= X.dtype)
                        incumbent_list_y = np.empty((0, 1), dtype = Y.dtype)
                    x_star = X[Y.argmin()]
                    y_star = Y.min()
                    incumbent_list_x = np.append(incumbent_list_x,np.array([x_star]),axis=0)
                    incumbent_list_y = np.append(incumbent_list_y,np.array([[y_star]]),axis=0)
                    
            if args.show_acq_method or args.show_obj_method or args.show_model_method:
                if bo.model_untrained:
                    first_vis = args_dict.copy()
                    del first_vis["show_acq_method"]
                    del first_vis["show_model_method"]
                    vis = Visualization(bo, new_x, X, Y, dest_folder, prefix="%03d_iteration" % (i,), **first_vis)
                else: 
                    vis = Visualization(bo, new_x, X, Y, dest_folder, prefix="%03d_iteration" % (i,), **args_dict)
            
        except IOError, e:
            break
        i += 1
    print incumbent_list_y, incumbent_list_x
    
    vis = Visualization(bo, incumbent_list_x=incumbent_list_x, incumbent_list_y=incumbent_list_y, dest_folder=dest_folder, prefix="incumbent_gap" , show_incumbent_gap=args.show_incumbent_gap)
     
        
        
        
