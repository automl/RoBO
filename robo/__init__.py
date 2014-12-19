import numpy as np
def bayesian_optimization_main(objective_fkt, acquisition_fkt, model, maximize_fkt, X_lower, X_upper,  maxN):
    for i in xrange(maxN):
        acquisition_fkt.model_changed()
        new_x = maximize_fkt(acquisition_fkt, X_lower, X_upper)
        new_y = objective_fkt(new_x)
        model.update(np.array(new_x), np.array(new_y))
    return model.getCurrentBestX(), model.getCurrentBest()
        
def bayesian_optimization(acquisition_fkt, model, maximize_fkt, X_lower, X_upper):
    model.train(np.array(new_x), np.array(new_y))
    return maximize_fkt(acquisition_fkt, X_lower, X_upper)