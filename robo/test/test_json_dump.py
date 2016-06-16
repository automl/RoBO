'''

Created on: June 5th, 2016
@author: Numair Mansur (numair.mansur@gmail.com)

'''
import unittest
import numpy as np
import george
from robo.maximizers.direct import Direct
from robo.acquisition.ei import EI
from robo.models.gaussian_process import GaussianProcess
from robo.task.synthetic_functions.levy import Levy

from robo.solver.base_solver import BaseSolver
from robo.models.base_model import BaseModel
from robo.task.base_task import BaseTask
from robo.acquisition.base_acquisition import BaseAcquisitionFunction
from robo.solver.bayesian_optimization import BayesianOptimization


class TestJsonMethods(unittest.TestCase):

    def test_json_base_solver(self):
        task = Levy()
        kernel = george.kernels.Matern52Kernel([1.0], ndim=1)
        model = GaussianProcess(kernel)
        ei = EI(model, task.X_lower, task.X_upper)
        maximizer = Direct(ei, task.X_lower, task.X_upper)
        solver = BayesianOptimization(acquisition_func=ei,
                          model=model,
                          maximize_func=maximizer,
                          task=task
                          )
        solver.run(1,X =None, Y=None)
        iteration = 0
        data = solver.get_json_data(it=iteration)
        assert data['iteration'] == iteration

    def test_json_base_model(self):
        model = BaseModel()
        model.X = None
        model.Y = None
        assert model.get_json_data()['X'] == None
        assert model.get_json_data()['Y'] == None
        model.X = np.random.rand(3,2)
        model.Y = np.random.rand(3,2)
        assert model.get_json_data()['X'] == model.X.tolist()
        assert model.get_json_data()['Y'] == model.Y.tolist()


    def test_json_base_task(self):
        X_lower = np.random.rand(3,2)
        X_upper = np.random.rand(3,2)
        opt = np.random.rand(3,2)
        fopt=np.random.rand(3,2)

        task = BaseTask(X_lower = X_lower,
                        X_upper = X_upper,
                        opt = opt,
                        fopt=fopt )

        data = task.get_json_data()
        assert data['opt']  == task.transform(opt).tolist()
        assert data['fopt'].tolist() == fopt.tolist()
        assert data['original_X_lower'] == X_lower.tolist()
        assert data['original_X_upper'] == X_upper.tolist()

    def test_json_base_acquisition(self):
        model = BaseModel()
        X_lower = np.random.rand(3,2)
        X_upper = np.random.rand(3,2)
        acquistion = BaseAcquisitionFunction(model=model,
                                             X_lower=X_lower,
                                             X_upper = X_upper)
        data = acquistion.get_json_data()
        assert data['type'] == 'robo.acquisition.base_acquisition'


if __name__ == "__main__":
    unittest.main()