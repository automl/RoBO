# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:35:39 2016

@author: christina
"""
from pymatbridge import Matlab
import numpy as np

def start_matlab():
        steps = 10        
        x = np.array([[-1,-1,-1],[1,2,3], [-6.7351, 1.0427], [-1,3,2], [1,1,1]])
        mlab = Matlab(executable='/home/christina/Matlab/bin/matlab')
        mlab.start()
        output = mlab.run_func('/home/christina/HIWI/Three_link_walker/simulation/walker_simulation.m', 
                               {'arg1': x[0], 'arg2': x[1], 'arg3': x[2], 'arg4': x[3],'arg5': x[4], 'arg6': steps})
        print(output)
        #answer = output['result']
        # in order to get the controller params
        #controller_params = answer['controller_params']
        #simul_output = answer['simul_output']
        mlab.stop()
        
#start_matlab()
x = np.array([[-1,-2,-3,-4,-5, -6, -7,-8,-9, -10,-11,-12, -13,-41,-15]]) 
print(x[0][:3]) 
print(x[0][3:6]) 
print(x[0][6:9])
print(x[0][9:12]) 
print(x[0][12:])    
#y = (x[1:1] - (5.1 / (4 * np.pi ** 2)) * x[:0] ** 2 + 5 * x[:0] / np.pi - 6) ** 2
#y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:0]) + 10
#print(y)
#print(y[:, np.newaxis])
