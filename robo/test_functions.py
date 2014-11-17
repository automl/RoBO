import math
import numpy as np
def branin(x):
    x1 = x[:,0]
    x2 = x[:,1]
    a = 1
    b = 5.1/(4*math.pi**2)
    c = 5/math.pi
    r = 6;
    s = 10;
    t = 1/(8*math.pi);
    f = (x2-b*x1*x1+c*x1-r)
    y = a*f*f+s*(1-t)*np.cos(x1)+s;
    return y

def branin2(x):
    c2 = np.empty(x[0].shape)
    c2.fill(12) 
    return branin(np.array([[x[0], c2]]))
 
    