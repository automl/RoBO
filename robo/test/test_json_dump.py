'''

Created on: June 5th, 2016
@author: Numair Mansur (numair.mansur@gmail.com)

'''
import unittest
import numpy as np

from robo.solver.base_solver import BaseSolver
from robo.models.base_model import BaseModel
from robo.task.base_task import BaseTask
from robo.acquisition.base_acquisition import BaseAcquisitionFunction


class TestJsonMethods(unittest.TestCase):

    def test_json_base_solver(self):
        solver = BaseSolver()
        assert True

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
        assert True # fix this !


if __name__ == "__main__":
    unittest.main()