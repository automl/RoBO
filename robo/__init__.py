import random
import os
import errno
import numpy as np
import shutil
here = os.path.abspath(os.path.dirname(__file__))
class BayesianOptimization(object):
    def __init__(self, acquisition_fkt, model, maximize_fkt, X_lower, X_upper, dims, objective_fkt=None):
        self.objective_fkt = objective_fkt
        self.acquisition_fkt = acquisition_fkt
        self.model = model
        self.maximize_fkt = maximize_fkt
        self.X_lower = X_lower
        self.X_upper = X_upper
        self.dims = dims
         
    def run(self, num_iterations=10, save_dir=None, X=None, Y=None, overwrite=True):
        """
        save_dir: 
            save to save_dir after each iteration
        overwrite:
            True: data present in save_dir will be deleted.
            False: data present will be loaded an the run will continue
        X, Y:
            Initial observations. They are optional. If a run continues
            these observations will be overwritten by the load
        """
        def _onerror(dirs, path, info):
            if info[1].errno != errno.ENOENT:
                raise
        if overwrite:
            shutil.rmtree(save_dir, onerror=_onerror)
            
        if num_iterations > 1:
            new_x, old_best_x, old_best_y, X, Y = self.run(num_iterations=num_iterations-1, save_dir=save_dir, X=X, Y=Y, overwrite=False) 
            new_y = np.array(self.objective_fkt(np.array(new_x)))
            if X is not None and Y is not None:
                X = np.append(X, new_x, axis=0)
                Y = np.append(Y, new_y, axis=0)
            else:
                X = new_x
                Y = new_y
            new_x = self.get_next_x(X, Y)
        else:        
            new_x = self.get_next_x(X, Y)
            
        if save_dir != None:
            self.save_iteration(save_dir, X, Y, new_x);
        
        return new_x, self.model.getCurrentBestX(), self.model.getCurrentBest(), X, Y
                
        
    def get_next_x(self, X=None, Y=None):
        if X is not None and Y is not None:
            self.model.train(X, Y)
            self.acquisition_fkt.update(self.model)
            return self.maximize_fkt(self.acquisition_fkt, self.X_lower, self.X_upper)
        else:
            X = np.empty((1, self.dims)) 
            for i in range(self.dims):
                X[0,i] = random.random() * (self.X_upper[i] - self.X_lower[i]) + self.X_lower[i];
            return np.array(X)
    
    def save_iteration(self, save_dir, X, Y, new_x):
        try:
            os.makedirs(save_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        
        