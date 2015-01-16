class ModelVisualization(object):
    pass

class AcquisitionVisualization(object):
    pass

class Visualization(object):
    def __init__(self, bayesian_opt, new_x, X, Y, dest_folder, prefix="", acq_method = None):
        if bayesian_opt.dims > 1 and acq_method is not None:
            raise AttributeError("acquisition function can only be visualized if the objective funktion has only one dimension")
        self.nrows = 1
        self.ncols = reduce(lambda x, y : x + 1 if y is not None else x, [0, acq_method])
        self.prefix = prefix
        num = 1
        self.fig = plt.figure()
        if acq_method:
            self.acquisition_fkt = bayesian_opt.acquisition_fkt
            self.plot_acquisition_fkt(num)

    
    def plot_model(self, model):
        pass
    
    def plot_acquisition_fkt(self, num):
            ax = fig.add_subplot(self.nrows, self.ncols, num)
            """
            c1 = np.reshape(plotting_range, (obj_samples, 1))
            c2 = acquisition_fkt(c1)
            c2 = c2*50 / np.max(c2)
            c2 = np.reshape(c2,(obj_samples,))
            ax.plot(plotting_range,c2, 'r')
            ax.plot(plotting_range, branin_result, 'black')
            fig.savefig("%s/tmp/np_%s.png"%(here, i), format='png')
            fig.clf()
            plt.close()
            """
    
    def plot_improvement(self, observations):
        pass
        