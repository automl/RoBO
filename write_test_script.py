
import matplotlib; matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt;
import os
import sys
import errno
import subprocess
import random
import numpy as np
from robo.test_functions import branin
try:
    import cpickle as pickle
except:
    import pickle

objectives = (("branin", np.array(((-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475))), 0.397887),)
maximizers = ("stochastic_local_search",)
n = 200
folder_prefix = "/tmp/robo_run/"

parameter_setups = {
    #"EI" :         [('0.3',)],
    #"PI" :         [('0.3',)],
    #"LogEI" :     [('0.0',), ('0.3',), ('1.5',)],
    #"UCB" :        [('1.0',)],
    "Entropy" :     [('20', '300')],
    "EntropyMC" :     [('20', '300', '2000')],
}

colors = ["#000000","#ff0000", "#000077", "#0000ff", "#007700",
          "#00ff00", "#770000",  "#777700",
          "#ffff00", "#770077", "#ff00ff", "#007777",
          "#00ffff", "#ff7700", "#ff0077", "#00ff77"]

seed = random.random()
cmd_template = "robo_examples --overwrite -a %(acquisition_fkt)s -p %(acq_param)s -m GPy --seed %(seed)s -e %(maximize)s -n %(n)s -o %(objective)s %(target_folder)s  &"
folder_name_layout = "%(acquisition_fkt)s_%(acq_param)s_%(objective)s"

num_runs = reduce(lambda c, l: c + len(l), parameter_setups.values(), 0) * len(objectives) * len(maximizers)
print "will do %s runs" % num_runs

def create_dir(folder):
    if folder is not None:
        try:
            os.makedirs(folder)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

x_dist_mins = []
y_dist_mins = []
fig = plt.figure()
ax_current_believe_y_dist = fig.add_subplot(212)
ax_current_believe_dist = fig.add_subplot(211)

ax_current_believe_y_dist.set_title("|f(x) - f(x*)|")
ax_current_believe_y_dist.set_yscale('log')
ax_current_believe_dist.set_title("|x - x*|")
ax_current_believe_dist.set_yscale('log')
 
color_idx = 0
for objective in objectives:
    for maximizer in maximizers:
        for acquisition_function, params in parameter_setups.items():
            for param_setup in params:
                folder_name = folder_prefix + folder_name_layout % dict(acquisition_fkt=acquisition_function, acq_param="_".join(param_setup), objective=objective[0])
                create_dir(folder_name)
                if sys.argv[1] == "run":
                    cmd = cmd_template % dict(acquisition_fkt=acquisition_function, acq_param=" ".join(param_setup), maximize=maximizer, objective=objective[0], n=n, target_folder=folder_name, seed=seed)
                    print cmd
                    subprocess.call(cmd, shell=True)
                   
                if sys.argv[1] == "show":
                    x_dist_min = np.empty((n,))
                    x_dist_min.fill(np.nan)
                    y_dist_min = np.empty((n,))
                    y_dist_min.fill(np.nan)
                    inc_x_dists_min = np.empty((n,))
                    inc_x_dists_min.fill(np.nan)
                    inc_x_dist = np.inf
                    for i in range(1, n + 1):
                        
                        try:
                            new_x, X, Y, best_guess = pickle.load(open(folder_name + "/%03d" % (i,) + "/observations.pickle", "rb"))
                            if X is None:
                                continue
                            best_guess_y = branin(best_guess[None,:])
                            
                            print best_guess_y
                            x_dist = np.empty((objective[1].shape[0],))
                            
                            for j in range(objective[1].shape[0]):
                                if np.linalg.norm(X[-1] - objective[1][j], 2) < inc_x_dist:
                                    inc_x_dist = np.linalg.norm(X[-1] - objective[1][j], 2)
                                x_dist[j] = np.linalg.norm(best_guess[0] - objective[1][j], 2)
                            x_dist_min[i - 2] = x_dist.min()
                            y_dist_min[i-2] =  best_guess_y - objective[2]
                            inc_x_dists_min[i - 2] = inc_x_dist
                        except Exception,e:
                            print e
                            break
                    x_dist_mins.append(x_dist_min)
                    y_dist_mins.append(y_dist_min)  
                    print np.arange(1, i - 1).shape, inc_x_dists_min[0:i - 2].shape
                    #ax_current_believe_y_dist.plot(np.arange(1, i - 1), x_dist_min[0:i - 2], color=colors[color_idx], label=acquisition_function)
                    ax_current_believe_dist.plot(np.arange(1, i - 1), x_dist_min[0:i - 2], color=colors[color_idx], label=acquisition_function)
                    ax_current_believe_y_dist.plot(np.arange(1, i - 1), y_dist_min[0:i - 2], color=colors[color_idx], label=acquisition_function)
                    #ax_incumbent_dist.plot(np.arange(1, i - 1), inc_x_dists_min[0:i - 2], color=colors[color_idx], label=acquisition_function)
                    color_idx += 1
                    print acquisition_function, "\n------------------------------------\n", x_dist_min, "\n", "~"*40, "\n", y_dist_min
                    plt.legend()
if sys.argv[1] == "show":
    plt.show(block=True)
