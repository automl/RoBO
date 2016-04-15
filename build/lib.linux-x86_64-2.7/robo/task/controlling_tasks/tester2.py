# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:35:39 2016

@author: christina
"""
from pymatbridge import Matlab
import numpy as np

def start_matlab():
        #q_minus = np.array([np.pi/2-np.pi/8, -2*(np.pi/2-np.pi/8), -(np.pi/6-np.pi/8)])
        q_minus = np.array([ 0.38987359,  0.10226458,  0.35467067])
        #dq_minus = np.multiply(1.23,[-1, 2, 1])
        dq_minus = np.array([0.92609643,  0.79602321,  0.20908526])

        steps = 10
        mlab = Matlab(executable='/home/christina/Matlab/bin/matlab')
        mlab.start()
        output = mlab.run_func('/home/christina/HIWI/Three_link_walker/simulation/walker_simulation.m', {'arg1': q_minus, 'arg2': dq_minus, 'arg3': steps})
        #print(output)
        answer = output['result']
        print(answer)
        controller_params = answer['controller_params']
        
        #simul_output = answer['simul_output']
        print(controller_params)
        mlab.stop()
        
start_matlab()
#x = np.array([-5, 0])        
#y = (x[1:1] - (5.1 / (4 * np.pi ** 2)) * x[:0] ** 2 + 5 * x[:0] / np.pi - 6) ** 2
#y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:0]) + 10
#print(y)
#print(y[:, np.newaxis])
