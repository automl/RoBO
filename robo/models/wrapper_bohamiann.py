import numpy as np
import torch

from pybnn.bohamiann import Bohamiann
from pybnn.multi_task_bohamiann import MultiTaskBohamiann

from robo.models.base_model import BaseModel


def get_default_network(input_dimensionality: int) -> torch.nn.Module:
    class AppendLayer(torch.nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = torch.nn.Parameter(torch.FloatTensor(1, 1))
            else:
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            torch.nn.init.constant_(module.bias, val=np.log(1e-2))
        elif type(module) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            torch.nn.init.constant_(module.bias, val=0.0)

    return torch.nn.Sequential(
        torch.nn.Linear(input_dimensionality, 50), torch.nn.Tanh(),
        torch.nn.Linear(50, 50), torch.nn.Tanh(),
        torch.nn.Linear(50, 1),
        AppendLayer()
    ).apply(init_weights)


class WrapperBohamiann(BaseModel):

    def __init__(self, get_net=get_default_network, lr=1e-2, use_double_precision=True, verbose=True):
        """
        Wrapper around pybnn Bohamiann implementation. It automatically adjusts the length by the MCMC chain,
        by performing 100 times more burnin steps than we have data points and sampling ~100 networks weights.

        Parameters
        ----------
        get_net: func
            Architecture specification

        lr: float
           The MCMC step length

        use_double_precision: Boolean
           Use float32 or float64 precision. Note: Using float64 makes the training slower.

        verbose: Boolean
           Determines whether to print pybnn output.
        """

        self.lr = lr
        self.verbose = verbose
        self.bnn = Bohamiann(get_network=get_net, use_double_precision=use_double_precision)

    def train(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.bnn.train(X, y, lr=self.lr,
                       num_burn_in_steps=X.shape[0] * 100,
                       num_steps=X.shape[0] * 100 + 10000, verbose=self.verbose)

    def predict(self, X_test):
        return self.bnn.predict(X_test)


class WrapperBohamiannMultiTask(BaseModel):

    def __init__(self, n_tasks=2, lr=1e-2, use_double_precision=True, verbose=False):
        """
        Wrapper around pybnn Bohamiann implementation. It automatically adjusts the length by the MCMC chain,
        by performing 100 times more burnin steps than we have data points and sampling ~100 networks weights.

        Parameters
        ----------
        get_net: func
            Architecture specification

        lr: float
           The MCMC step length

        use_double_precision: Boolean
           Use float32 or float64 precision. Note: Using float64 makes the training slower.

        verbose: Boolean
           Determines whether to print pybnn output.
        """

        self.lr = lr
        self.verbose = verbose
        self.bnn = MultiTaskBohamiann(n_tasks,
                                      use_double_precision=use_double_precision)

    def train(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.bnn.train(X, y, lr=self.lr, mdecay=0.01,
                       num_burn_in_steps=X.shape[0] * 500,
                       num_steps=X.shape[0] * 500 + 10000, verbose=self.verbose)

    def predict(self, X_test):
        return self.bnn.predict(X_test)
