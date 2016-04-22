from pymatbridge import Matlab
import numpy as np
from robo.task.base_task import BaseTask


class Walker(BaseTask):
    
    def __init__(self):
        '''       
        This task is based on the Walker simulation which was originally used
        for experiments in [1].
        The idea of using it was inspired by [2].
        
        The simulation code examples can be found in 
        http://web.eecs.umich.edu/~grizzle/biped_book_web/
        [three-link-walker simulation]
        

        [1] E. Westervelt and J. Grizzle. Feedback Control of Dynamic
        Bipedal Robot Locomotion. Control and Automation Series.
        CRC PressINC, 2007. ISBN 9781420053722.
        
        [2] Bobak Shahriari, Ziyu Wang, Matthew W. Hoffman, 
        Alexandre Bouchard-Cote and Nando de Freitas. An Entropy Search 
        Portfolio for Bayesian Optimization. University of Oxford, 2014.
        "http://arxiv.org/abs/1406.4625"
        
        The controller parameters are given directly without being calculated 
        (like in walker_v2.py)
        '''
        X_lower = np.array([-5,-5,-5,-5,-5,-5,-5,-5,-5,-5])
        X_upper = np.array([5,5,5,5,5,5,5,5,5,5])        
        super(Walker, self).__init__(X_lower, X_upper)
        
    def objective_function(self, x):
        '''
        matlabpath: we need the path to matlab since we need to run it.
        
        IMPORTANT: walker_simulation.m must be included in the simulation file
        of three-link-walker in order to work. So simply modify the path below.
        '''
        matlabpath = '/path/to/Matlab/bin/matlab'
        steps = 10  # can be changed       
        mlab = Matlab(executable= matlabpath)
        mlab.start()
        output = mlab.run_func('walker_simulation.m', 
                               {'arg1': np.asmatrix(x[0][:2]).T, 'arg2': np.asmatrix(x[0][2:4]).T, 
                               'arg3': np.asmatrix(x[0][4:6]).T, 'arg4': np.asmatrix(x[0][6:8]).T,
                               'arg5':np.asmatrix(x[0][8:]).T, 'arg6': steps})
                               

        answer = output['result']
        simul_output = answer['speed']
        mlab.stop()
        
        return simul_output[:, np.newaxis]
        
        
    def objective_function_test(self, x):
       return self.objective_function(x)
