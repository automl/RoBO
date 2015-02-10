RoBO - a Robust Bayesian Optimization framework.
================================================

Installation
~~~~~~~~~~~~

Using a virtual environment
---------------------------

create a environment with system packages enabled. This will install pip and python executables with different path settings under


	.. code:: 
	 
	   mkdir ~/envs
	   virtualenv --system-site-packages ~/envs/robo
	  
then you activate the environment with

	.. code:: 
	 
	   source ~/envs/robo/bin/activate
	   
Install the editable version of robo with

	.. code:: 
	 
	   pip install -e <path to the robo repository>
	   
Then there will be a new executable under ~/envs/robo/bin/robo_main, that you can call. It will produce some images under <path to the robo repository>/tmp/


