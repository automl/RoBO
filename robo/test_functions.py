import math
import numpy as np
def branin(x):
    x1 = x[0]
    x2 = x[1]
    a = 1
    b = 5.1/(4*math.pi**2)
    c = 5/math.pi
    r = 6;
    s = 10;
    t = 1/(8*math.pi);
    f = (x2-b*x1*x1+c*x1-r)
    #print a*f*f
    return a*f*f+s*(1-t)*np.cos(x1)+s;

def branin2(x):
    return [branin([x[0], 12])] 
if __name__ == "__main__":
    x1 = np.random.uniform(-3.,3.,(20,1))
    x2 = np.random.uniform(-3.,3.,(20,1))
    obj_samples=5
    c1 = np.linspace(-8, 19, num=obj_samples)
    
    c2 = np.empty((obj_samples,))
    c2.fill(12)
    print(branin([c1,c2]))
    