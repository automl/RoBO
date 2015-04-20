from sklearn.gaussian_process import GaussianProcess

from robo.models.base_model import BaseModel


class SkLearnGP(BaseModel):
    """
     This class is a wrapper around the scikit-learn GP implementation
    """
    def __init__(self):
        self.gp = GaussianProcess()
        super(SkLearnGP, self).__init__()

    def train(self, X, y):
        self.gp.fit(X, y)
        super(SkLearnGP, self).train(X, y)

    def predict(self, X):
        mean, var = self.gp.predict(X, eval_MSE=True)
        return mean, var

    def getCurrentBestX(self):
        return super(SkLearnGP, self).getCurrentBestX()
