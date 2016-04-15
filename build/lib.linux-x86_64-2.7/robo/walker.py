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
        X_lower = np.array([[0,0,0],[0,0,0]])
        X_upper = np.array([[1,1,1],[1,1,1]])        
        super(Walker, self).__init__(X_lower, X_upper)
        
    def objective_function(self, x):
        '''
        You need pymatbridge library in order for it to work.
        '''
        q_minus = x[0]
        dq_minus = x[1]
        steps = 10        
        
        mlab = Matlab(executable='/home/christina/Matlab/bin/matlab')
        mlab.start()
        output = mlab.run_func('/home/christina/HIWI/Three_link_walker/simulation/walker_simulation.m', {'arg1': q_minus, 'arg2': dq_minus, 'arg3': steps})
        answer = output['result']
        # in order to get the controller params
        #controller_params = answer['controller_params']
        simul_output = answer['simul_output']
        mlab.stop()
        
        return simul_output[:, np.newaxis]
        
        
    def objective_function_test(self, x):
       return self.objective_function(x)