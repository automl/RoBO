function output = walker_simulation2(args)
% filename: walker_simulation2
% purpose: This file contains all the setup necessary to run a simulation of
%          the three-link biped as a RoBo task.
% The simulator which is called here is a software held by Professor Jessy Grizzle, University of Michigan, Professor Eric Westervelt, The Ohio State University, and Mr. Ben Morris, University of Michigan.

% where to look for needed m-files 
addpath ../functions_manual
addpath ../functions_auto_gen
addpath ../simulation

% choose desired state of position and velocity at impact
q_minus = args.arg1';
dq_minus = args.arg2';


% generate the controller parameters which depend on these desired states
[a0,a1,a3,a4] = poly_coeff(q_minus,dq_minus);
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

steps=args.arg3;
do_animation=0;
draw_graphs=0;

[x,t]=full_simul(x0,a,theta_minus,steps,do_animation,draw_graphs);

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
