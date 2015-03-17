import os
import errno
import subprocess

objectives = ("branin",)
maximizers = ("stochastic_local_search",)
n = 200
folder_prefix = "/tmp/robo_run/"

parameter_setups = {
	"EI" : 		[('0.0',), ('0.3',), ('1.5',)],
	"PI" : 		[('0.0',), ('0.3',), ('1.5',)],
	"LogEI" : 	[('0.0',), ('0.3',), ('1.5',)],
        "UCB" : 	[('0.3',), ('0.6',), ('1.0',)],
        #"Entropy" : 	[('50', '200'), ('80', '500')],
        #"EntropyMC" : 	[('50', '200', '1200'), ('80', '500', '2000')],
}

cmd_template = "robo_examples --overwrite -a %(acquisition_fkt)s -p %(acq_param)s -m GPy -e %(maximize)s -n %(n)s -o %(objective)s %(target_folder)s  &"
folder_name_layout = "%(acquisition_fkt)s_%(acq_param)s_%(objective)s"

num_runs = reduce(lambda c, l: c+len(l), parameter_setups.values(), 0)*len(objectives)*len(maximizers)
print "will do %s runs" % num_runs

def create_dir(folder):
    if folder is not None:
        try:
            os.makedirs(folder)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

for objective in objectives:
    for maximizer in maximizers:
        for acquisition_function, params in parameter_setups.items():
            for param_setup in params:
                folder_name = folder_prefix + folder_name_layout % dict(acquisition_fkt= acquisition_function, acq_param="_".join(param_setup), objective=objective)
                create_dir(folder_name)
                log_file = folder_name+".log"
                lf = open(log_file, "w+")
              
                cmd = cmd_template % dict(acquisition_fkt= acquisition_function, acq_param=" ".join(param_setup), maximize=maximizer, objective=objective, n=n, target_folder=folder_name)
                print cmd
                subprocess.call(cmd,  shell=True)

