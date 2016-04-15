function output = walker_simulation(args)
% filename: test_full_simul
% purpose: This file contains all the setup necessary to run a simulation of
%          the three-link biped.  Note that this simulator will run any 
%          given controller configuration, it need not be stable or
%          physically feasible (ground penetrations are allowed).

% where to look for needed m-files 
addpath ../functions_manual
addpath ../functions_auto_gen
addpath ../simulation

% choose desired state of position and velocity at impact (example)
%q_minus = [pi/2-pi/8 -2*(pi/2-pi/8) -(pi/6-pi/8)]';
%dq_minus = [-1 2 1]'.*1.35;

q_minus = args.arg1';
dq_minus = args.arg2';


% generate the controller parameters which depend on these desired states
% we don't need it if we want to use parameters directly instead of calculating them (otherwise uncomment)
[a0,a1,a3,a4] = poly_coeff(q_minus,dq_minus);
% choose what the 'free parameter' of the controller should be
% note that the parameter 'a' does not contain all the information
% necessary to describe the zero dyamics.  The values of 'theta_minus' 
% must also % be passed everywhere

%a2 = [-6.7351 1.0427]'; % this gives an ice skater style motion
a2=(a1+a3)/2;
a = [a0 a1 a2 a3 a4];
output.controller_params = a;

% find inital conditions for the biped
% x0 is on the zero dynamics but not on the limit cycle
theta_minus = q_to_theta(q_minus);
dtheta_minus = -dq_minus(1);
x0 = find_ic(a,theta_minus,dtheta_minus);

output.theta_minus = theta_minus;
output.dtheta_minus = dtheta_minus;
output.x0 = x0;

%----------------------------------------------------------------------------------------------
% note: these same initial conditions may be found by the following
% q_plus=delta_q(q_minus);
% dq_plus=delta_dq(dq_minus);
% x0=[q_plus dq_plus];
% these two methods are only equivalent when q_minus and dq_minus are fully in the zero dynamics
%-----------------------------------------------------------------------------------------------

% execute the simulation 
% parameters are as follows:  
%    steps = # of impacts to simulate (if impacts do not occur after some time, simulation stops and outputs a message.  See 'full_simul.m')
%    do_animation = 1 if animation is desired
%                 = 0 for no animation (may be used when use user is only interested in output data)
%    draw_graphs = 1 if graphs are desired ()
%                = 0 for no graphs

steps=args.arg3;
do_animation=0;
draw_graphs=0;

[x,t]=full_simul(x0,a,theta_minus,steps,do_animation,draw_graphs);
simul_output = [x,t];
output.simul_output = simul_output;
end
