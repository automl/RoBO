from robo.acquisition.base import AcquisitionFunction
 
class UCB(AcquisitionFunction):
    long_name = "Upper Confidence Bound" 
    def __init__(self, model, par=1.0, **kwargs):
        self.model = model
        self.par = par
    def __call__(self, X, Z=None, **kwargs):
        mean, var = self.model.predict(X, Z)
        return -mean + self.par * np.sqrt(var)
    def update(self, model):
        self.model = model