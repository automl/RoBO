% setup for Entropy Search using the physical experiment function handle

% in these two lines, you have to add the paths to 
run H:/Documents/MATLAB/gpml-matlab-v3.1-2010-09-27/startup.m  % gpml toolbox
addpath H:/min_factor/logsumexp/  % logsumexp package
addpath H:/min_factor/tprod/      % tprod package

% set up prior belief
N               = 3; % number of input dimensions
in.covfunc      = {@covSEard};       % GP kernel
in.covfunc_dx   = {@covSEard_dx_MD}; % derivative of GP kernel. You can use covSEard_dx_MD and covRQard_dx_MD if you use Carl's & Hannes' covSEard, covRQard, respectively.
hyp.cov         = log([ones(N,1);1]); % hyperparameters for the kernel
hyp.lik         = log([1e-3]); % noise level on signals (log(standard deviation));
in.hyp          = hyp;  % hyperparameters, with fields .lik (noise level) and .cov (kernel hyperparameters), see documentation of the kernel functions for details.

% should the hyperparameters be learned, too?
in.LearnHypers  = true; % yes.
in.HyperPrior   = @SEGammaHyperPosterior;

% constraints defining search space:
in.xmin         = [-1,-1,-1]; % lower bounds of rectangular search domain
in.xmax         = [1,1,1]; % upper bounds of rectangular search domain
in.MaxEval      = 5;    % Horizon (number of evaluations allowed)

% objective function:
in.f            = @(x) PhysicalExperiment(x); % handle to objective function

result = EntropySearch(in) % the output is a struct which contains GP datasets, which can be evaluated with the gpml toolbox.