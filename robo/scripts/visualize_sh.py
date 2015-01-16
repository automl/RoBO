import argparse
import os
import robo
import errno
from robo.visualization import Visualization
def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description='Visualize a robo run', 
                                     prog='robo_visualize')
                                     #formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source_folder', metavar="SOURCE_FOLDER", type=str)#, required=True)
    parser.add_argument('dest_folder',   metavar="DESTINATION_FOLDER", type=str)#, required=True)
    parser.add_argument('-a', '--acq', metavar="METHOD", default=None, 
                        help='Choose a method to visualize the acquisition fkt', 
                        dest="acq_method", choices=('subplot',))
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
            vis = Visualization(bo, new_x, X, Y, dest_folder, prefix="%03d_"%(i,), **args.__dict__) 
        except Exception, e:
            print e
            break;
        i += 1
    
        