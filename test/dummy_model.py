import numpy as np

from robo.models.base_model import BaseModel


class DemoModel(BaseModel):

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
        return np.ones(X_test.shape[0]) * self.m, np.ones(X_test.shape[0]) * self.v

    @BaseModel._check_shapes_train
    def train(self, X, y):
        self.X = X
        self.y = y

        self.m = np.mean(y)
        self.v = np.var(y)

    def predictive_gradients(self, Xnew):
        pass


class DemoQuadraticModel(BaseModel):

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
        y = np.sum((0.5 - X_test) ** 2, axis=1)
        return y, np.ones(X_test.shape[0]) * 0.001

    @BaseModel._check_shapes_train
    def train(self, X, y):
        self.X = X
        self.y = y
