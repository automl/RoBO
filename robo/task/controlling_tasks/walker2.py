from pymatbridge import Matlab
import numpy as np
from robo.task.base_task import BaseTask


class Walker2(BaseTask):
    
    def __init__(self):
        '''       
        This task is based on the Walker simulation which was originally used
        for experiments in [1].
        The idea of using it was inspired by [2].
        
        The simulation code examples can be found in 
        http://web.eecs.umich.edu/~grizzle/biped_book_web/
        [three-link-walker simulation]
        
        The whole MATLAB code can also be found in 
        http://web.eecs.umich.edu/~grizzle/CDC2003Workshop/

        [1] E. Westervelt and J. Grizzle. Feedback Control of Dynamic
        Bipedal Robot Locomotion. Control and Automation Series.
        CRC PressINC, 2007. ISBN 9781420053722.
        
        [2] Bobak Shahriari, Ziyu Wang, Matthew W. Hoffman, 
        Alexandre Bouchard-Cote and Nando de Freitas. An Entropy Search 
        Portfolio for Bayesian Optimization. University of Oxford, 2014.
        "http://arxiv.org/abs/1406.4625"

        Parameters
        ----------
        q_minus : desired state of position
        dq_minus : desired velocity at impact
        steps: # of impacts to simulate
        
        The controller parameters are calculated via q_minus and dq_minus
        '''
        X_lower = np.array([-1,-1,-1,-1,-1,-1])
        X_upper = np.array([3,3,3,3,3,3])        
        super(Walker2, self).__init__(X_lower, X_upper)
        
    def objective_function(self, x):
        '''
        You need pymatbridge library in order for it to work.
        
        IMPORTANT: walker_simulation2.m must be included in the simulation file
        of three-link-walker in order to work. So simply modify the path below.
        '''
        q_minus = x[0][:3]
        dq_minus = x[0][3:]
        steps = 10        
        
        mlab = Matlab(executable='/path/to/Matlab/bin/matlab')
        mlab.start()
        output = mlab.run_func('walker_simulation2.m', {'arg1': q_minus, 'arg2': dq_minus, 'arg3': steps})
        answer = output['result']
        simul_output = answer['speed']
        mlab.stop()
        
        return simul_output[:, np.newaxis]
        
        
    def objective_function_test(self, x):
       return self.objective_function(x)
