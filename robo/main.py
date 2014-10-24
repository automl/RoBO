import os
import random
import errno
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pb
#pb.ion()
import numpy as np
import GPy
import DIRECT
from test_functions import branin as test_f
here = os.path.abspath(os.path.dirname(__file__))

def acquisition_fkt(x, user_data):
    mean, var, _025pm, _975pm = user_data["gp"].predict(x)#,  full_cov=True)
    y = mean -  np.sqrt(np.abs(var))
    if "acq_max_y" not in user_data.keys():
        user_data["acq_max_y"] = max(y)
    else:
        user_data["acq_max_y"] = max(user_data["acq_max_y"],max(y),)
    if "acq_min_y" not in user_data.keys():
        user_data["acq_min_y"] = min(y)
    else:
        user_data["acq_min_y"] = min(user_data["acq_min_y"],min(y),)
    return y, 0

def optimized_acquisition(gp, min_x, max_x):
    user_data={"gp":gp, "acq_data":None}
    x, fmin, ierror = DIRECT.solve(acquisition_fkt, l=[min_x], u=[max_x], maxT=2000, maxf=2000, user_data=user_data)
    return x, -fmin, user_data
    
def main(samples=10, min_x=-8, max_x=19):
    X = np.empty((samples, 1))
    Y = np.empty((samples, 1))
    obj_samples = 70
    c1 = np.linspace(min_x, max_x, num=obj_samples)
    c2 = np.empty((obj_samples,))
    c2.fill(12)
    c3 = test_f([c1, c2])
    d1 = np.reshape(c1, (obj_samples,1))
    kernel = GPy.kern.rbf(input_dim=1, variance=70**2, lengthscale=5.)
    x = random.random() * (max_x - min_x) + min_x
    X[0, 0] = x
    user_data = None 
    # print X[i,0]
    Y[0, 0] = test_f([X[0, 0], 12])
    for i in xrange(0, samples):
        _X = X[0:i + 1, 0:1]
        _Y = Y[0:i + 1, 0:1]
        m = GPy.models.GPRegression(_X, _Y, kernel)
        m.unconstrain('')
        m.constrain_positive('.*rbf_variance')
        m.constrain_bounded('.*lengthscale', 1., 10.)
        m.constrain_fixed('.*noise', 0.0025)
        if i+1 < samples:
            x, y, user_data = optimized_acquisition(m, min_x, max_x)
            X[i+1, 0] = x
            Y[i+1, 0] = test_f([X[i+1, 0], 12])
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        m.plot(ax=ax, plot_limits=[min_x, max_x])
        xlim_min, xlim_max, ylim_min, ylim_max =  ax.axis()
        ax.set_ylim(min(0, ylim_min), max(230, ylim_max))
        
        ax.plot(c1, c3, 'r')
        xlim_min, xlim_max, ylim_min, ylim_max =  ax.axis()
        ud = {"gp":m, "acq_data":None}
        d2 = acquisition_fkt(d1, ud)[0].reshape((obj_samples,))
        d2 *= -1
        #d2 = (((d2-ud["acq_min_y"])/(float(ud["acq_max_y"]) - float(ud["acq_min_y"])))*50.0 )
        plt.plot(c1, d2, "y")
        plt.plot(x,y, "b+")#(y-user_data["acq_min_y"] )/(user_data["acq_max_y"]-float(user_data["acq_min_y"] )) *50,"b+")
         
        fig.savefig("%s/tmp/save_plot_%s.png"%(here, i), format='png')
        fig.clf()
        plt.close()

if __name__ == "__main__":
    
    try:
        os.makedirs("%s/tmp/"%here)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    main()
