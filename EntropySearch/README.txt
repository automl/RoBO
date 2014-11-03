VERSION 1.0 -- released May 2012

This is Matlab demonstration code for Entropy Search, as described in http://arxiv.org/abs/1112.1217

You will need the following external code packages

* the gpml toolbox by Carl Rasmussen and Hannes Nickisch: 
	http://www.gaussianprocess.org/gpml/code/matlab/doc/

* the logsumexp package 
	http://www.mathworks.com/matlabcentral/fileexchange/28899

* the tprod package
	http://www.mathworks.com/matlabcentral/fileexchange/16275-tprod-arbitary-tensor-products-between-n-d-arrays


* the Matlab optimization toolbox. If you do not have this, you can go through the code and replace calls to fmincon with fminbnd (much less efficient), or with a call to minimize.m (which you can get from http://www.gaussianprocess.org/gpml/code/matlab/util/minimize.m). But note that minimize does not automatically handle linear constraints. You can implement those naively by changing function handles such that they return +inf whenever evaluated outside the bounds.


Having installed all those packages, you should be able to call

EntropySearch(in), where

in.covfunc      = {@covSEard};       % GP kernel
in.covfunc_dx   = {@covSEard_dx_MD}; % derivative of GP kernel. You can use covSEard_dx_MD and covRQard_dx_MD if you use Carl's & Hannes' covSEard, covRQard, respectively.
in.hyp          = hyp;  % hyperparameters, with fields .lik (noise level) and .cov (kernel hyperparameters), see documentation of the kernel functions for details.
in.xmin         = xmin; % lower bounds of rectangular search domain
in.xmax         = xmax; % upper bounds of rectangular search domain
in.MaxEval      = H;    % Horizon (number of evaluations allowed)
in.f            = @(x) f(x) % handle to objective function

That handle @f is obviously the core part of the problem. If you use this method for actual experimental design, use the "PhysicalExperiment" function handle, which simply prompts for user input at the selected locations. An example script can be found in ExampleSetup.m


Philipp Hennig and Christian Schuler, 2011