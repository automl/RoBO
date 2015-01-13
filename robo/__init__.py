import random
import os
import numpy as np

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
         
    def run(self, num_iterations=10, save=True, save_dir=here+"/tmp", X=None, Y=None):
        """
        save: 
            save to save_dir after each iteration
        """
        if num_iterations > 1:
            new_x, old_best_x, old_best_y, X, Y = self.run(num_iterations=num_iterations-1, save=save, save_dir=save_dir, X=X, Y=Y) 
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
    
    
"""    
def bayesian_optimization_main(objective_fkt, acquisition_fkt, model, maximize_fkt, X_lower, X_upper, maxN):
    for i in xrange(maxN):
        acquisition_fkt.update(model)
        new_x = maximize_fkt(acquisition_fkt, X_lower, X_upper)
        new_y = objective_fkt(np.array(new_x))
        model.update(np.array(new_x), np.array(new_y))
    return model.getCurrentBestX(), model.getCurrentBest()
        
def bayesian_optimization(acquisition_fkt, model, maximize_fkt, X_lower, X_upper):
    model.train(np.array(new_x), np.array(new_y))
    return maximize_fkt(acquisition_fkt, X_lower, X_upper)
"""