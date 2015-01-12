import numpy as np
here = os.path.abspath(os.path.dirname(__file__))
class BayesianOptimization(object):
    def __init__(acquisition_fkt, model, maximize_fkt, X_lower, X_upper, objective_fkt=None):
        self.objective_fkt = objective_fkt
        self.acquisition_fkt = acquisition_fkt
        self.model = model
        self.maximize_fkt = maximize_fkt
        self.X_lower = X_lower
        self.X_upper = X_upper
         
    def run(self, update_model=False, num_iterations=10, save=True, save_dir=here+"/tmp", update_objective=True, X=None, Y=None):
        """
        update_model: 
            call model.update insteaf of model.train
        save: 
            save to save_dir after each iteration
        """
        
        if num_iterations > 1:
            return self.run(update_model=update_model, num_iterations=num_iterations-1, save=save, save_dir=save_dir, update_objective=update_objective)
        
        
        
    def get_next_x(self):
        model.train(np.array(new_x), np.array(new_y))
        return maximize_fkt(acquisition_fkt, X_lower, X_upper)
    
    
    
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

