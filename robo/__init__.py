import matplotlib; 
matplotlib.use('Agg')
import random
import os
import errno
import numpy as np
import shutil
try:
    import cpickle as pickle
except:
    import pickle
from robo.models import GPyModel
here = os.path.abspath(os.path.dirname(__file__))
class BayesianOptimizationError(Exception):
    LOAD_ERROR = 1
    SINGLE_INPUT_ONLY = 2
    def __init__(self, errno, *args, **kwargs):
        self.errno = errno
        Exception.__init__(self,*args, **kwargs)
        

class BayesianOptimization(object):
    """
        save_dir: 
            save to save_dir after each iteration
    """
    def __init__(self, acquisition_fkt=None, model=None, maximize_fkt=None, X_lower=None, X_upper=None, dims=None, objective_fkt=None, save_dir=None):
        self.enough_arguments = reduce(lambda a, b: a and b is not None, [True, acquisition_fkt, model, maximize_fkt, X_lower, X_upper, dims])
        if self.enough_arguments:
            self.objective_fkt = objective_fkt
            self.acquisition_fkt = acquisition_fkt
            self.model = model
            self.maximize_fkt = maximize_fkt
            
            self.X_lower = X_lower
            #if len(self.X_lower.shape) ==1:
            #     self.X_lower = self.X_lower[:,np.newaxis]
            self.X_upper = X_upper
            #if len(self.X_upper.shape) ==1:
            #    self.X_upper = self.X_upper[:,np.newaxis]
            self.dims = dims
            self.save_dir = save_dir
            if save_dir is not None:
                self.create_save_dir()
            self.model_untrained = True
        
        elif save_dir is not None:
            self.save_dir = save_dir
        else:
            raise ArgumentError()
        
    def init_last_iteration(self):
        max_iteration = self._get_last_iteration_number()
        iteration_folder = self.save_dir + "/%03d" % (max_iteration, )
        that = pickle.load(open(iteration_folder+"/bayesian_opt.pickle", "rb"))
        self.objective_fkt = that.objective_fkt
        self.acquisition_fkt = that.acquisition_fkt
        self.model = that.model
        self.maximize_fkt = that.maximize_fkt
        self.X_lower = that.X_lower
        self.X_upper = that.X_upper
        self.dims = that.dims
        return pickle.load(open(iteration_folder+"/observations.pickle", "rb"))        
    
    @classmethod
    def from_iteration(cls, save_dir, i):
        iteration_folder = save_dir + "/%03d" % (i, )
        that = pickle.load(open(iteration_folder+"/bayesian_opt.pickle", "rb"))
        
        if not isinstance(that, cls):
            raise BayesianOptimizationError(BayesianOptimizationError.LOAD_ERROR, "not a robo instance")
        new_x, X, Y = pickle.load(open(iteration_folder+"/observations.pickle", "rb"))
        return that, new_x, X, Y
        
    def create_save_dir(self):
        if self.save_dir is not None:
            try:
                os.makedirs(self.save_dir)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
                
    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        """
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
            
        if overwrite and self.save_dir:
            shutil.rmtree(self.save_dir, onerror=_onerror)
            self.create_save_dir()
            
        if num_iterations > 1:
            new_x, old_best_x, old_best_y, X, Y = self.run(num_iterations=num_iterations-1, X=X, Y=Y, overwrite=False) 
            new_y = np.array(self.objective_fkt(np.array(new_x)))
            if X is not None and Y is not None:
                X = np.append(X, new_x, axis=0)
                Y = np.append(Y, new_y, axis=0)
            else:
                X = new_x
                Y = new_y
            new_x = self.get_next_x(X, Y)
        else:        
            if X is None and Y is None and self.save_dir:
                try:
                    new_x, X, Y = self.init_last_iteration()
                except IOError as exception:
                    if not self.enough_arguments:
                        raise
                    new_x = self.get_next_x(X, Y)
            
        if self.save_dir != None:
            self.save_iteration(X, Y, new_x);
        
        return new_x, self.model.getCurrentBestX(), self.model.getCurrentBest(), X, Y
                
        
    def get_next_x(self, X=None, Y=None):
        if X is not None and Y is not None:
            self.model.train(X, Y)
            self.model_untrained = False
            self.acquisition_fkt.update(self.model)
            return self.maximize_fkt(self.acquisition_fkt, self.X_lower, self.X_upper)
        else:
            X = np.empty((1, self.dims)) 
            for i in range(self.dims):
                X[0,i] = random.random() * (self.X_upper[i] - self.X_lower[i]) + self.X_lower[i];
            return np.array(X)
    

    
    def _get_last_iteration_number(self):
        max_iteration = 0
        for i in os.listdir(self.save_dir):
            try:
                it_num = int(i)
                if it_num > max_iteration:
                    max_iteration = it_num
            except Exception, e:
                print e
        return max_iteration
    
    def save_iteration(self, X, Y, new_x):
        max_iteration = self._get_last_iteration_number()
        iteration_folder = self.save_dir + "/%03d" % (max_iteration+1, )
        os.makedirs(iteration_folder)
        
        pickle.dump(self, open(iteration_folder+"/bayesian_opt.pickle", "w"))
        pickle.dump([new_x, X, Y], open(iteration_folder+"/observations.pickle", "w"))
        
        
        
        
        
        