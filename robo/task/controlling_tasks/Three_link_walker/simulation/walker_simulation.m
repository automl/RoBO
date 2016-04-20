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
q_minus = [pi/2-pi/8 -2*(pi/2-pi/8) -(pi/6-pi/8)]';
dq_minus = [-1 2 1]'.*1.35;


% choose what the 'free parameter' of the controller should be
% note that the parameter 'a' does not contain all the information
% necessary to describe the zero dyamics.  The values of 'theta_minus' 
% must also % be passed everywhere

a0 = args.arg1;
a1 = args.arg2;
a2 = args.arg3;
a3 = args.arg4;
a4 = args.arg5;

a = [a0 a1 a2 a3 a4];

% find inital conditions for the biped
% x0 is on the zero dynamics but not on the limit cycle
theta_minus = q_to_theta(q_minus);
dtheta_minus = -dq_minus(1);
x0 = find_ic(a,theta_minus,dtheta_minus);

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

steps=args.arg6;
do_animation=0;
draw_graphs=0;

[x,t]=full_simul(x0,a,theta_minus,steps,do_animation,draw_graphs);

ang_vels = x(:,4:6);
ang2_vel = x(:,5);
output.ang2_vel = ang2_vel;

a1_rad = x(:,4);
a2_rad = x(:,5);
a3_rad = x(:,6);

% we have the angular velocities in rad/sec but in order to get the speed
% we first need m/sec --> w*r. And we only need the legs.
[r,m,Mh,Mt,L,g]=model_params;
a1 = a1_rad*r;
a2 = a2_rad*r;
a3 = a3_rad*r;

speed = (a1 + a2 + a3)/3;

output.speed = speed;

end
