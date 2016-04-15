clear *

% symb_model_biped_3dof   %% file name

% Copyright held by Jessy Grizzle (University of Michigan), Eric Westervelt (The Ohio State University)
% and Ben Morris (University of Michigan). December 4, 2003. 
%
%
% Use this code at your own risk. No liability or warranty provided. No consulting
% provided. If a publication results from the use of this code or its
% modification, you must include a citation to the source.
%
%
% Matlab code due to Ben Morris with contributions from Eric Westervelt and
% Jessy Grizzle
%
% The model trate3d here was studied in the paper J.W. Grizzle, Gabriel Abba, and Franck Plestan, 
% "Asymptotically Stable Walking for Biped Robots: Analysis via Systems with Impulse Effects,"
% IEEE T-AC, Volume 46, No. 1, January 2001, pp. 51-64. 
%
% The coordinates used here are shown in Figure fig_biped_newer_coordinates.jpg 
% which is located in this directory. These coordinates are
% not the same as the coordinates used in the above paper. For a model in
% the original coordinates of the paper, see in this directory the file 
% symb_dynamics_old_coordinates.m 
%
% For the original coordinates, see in this directory: fig_biped_old_coordinates.jpg
%
%
% Coordinate relationship:
% theta1=-q1+pi/2;
% theta2=-q1-q2-pi/2;
% theta3=-q1-q3+pi/2;
% 
%

% =================================
% symbolic modeling - three degree of freedom dynamics
% =================================

addpath ..\functions_manual

disp('Begin symbolic computation of dynamics for walking model');

syms q1 q2 q3 dq1 dq2 dq3 theta3_d real
syms g m Mh Mt r L  positive      

q=[q1 q2 q3].';
dq=[dq1 dq2 dq3].';

p1=[ 1/2*r*cos(q1), 1/2*r*sin(q1)]; % center of mass of stance leg
p2=[ r*cos(q1), r*sin(q1)]; % hip
p3=p2 + [1/2*r*cos(q1+q2), 1/2*r*sin(q1+q2)]; % center of mass of swing leg
p4=p2 + [L*cos(q1+q3), L*sin(q1+q3)]; % center of mass of Torso

PE=g*m*p1(2)+g*Mh*p2(2)+g*m*p3(2)+g*Mt*p4(2);
PE=simple(PE);

v1=jacobian(p1,q)*dq;
v2=jacobian(p2,q)*dq;
v3=jacobian(p3,q)*dq;
v4=jacobian(p4,q)*dq;

KE1=simplify((1/2)*m*v1.'*v1); % because this is a point mass model, there are no inertia terms to add in.
KE2=simplify((1/2)*Mh*v2.'*v2);
KE3=simplify((1/2)*m*v3.'*v3);
KE4=simplify((1/2)*Mt*v4.'*v4);

KE=simple(KE1+KE2+KE3+KE4);

%
% Model NOTATION: Spong and Vidyasagar, page 142, Eq. (6.3.12)
%                 D(q)ddq + C(q,dq)*dq + G(q) = B*tau + E2*F_external
%

G=jacobian(PE,q).';
G=simple(G);
D=simple(jacobian(KE,dq).');
D=simple(jacobian(D,dq));

syms C real
n=max(size(q));
for k=1:n
    for j=1:n
        C(k,j)=0;
        for i=1:n
            C(k,j)=C(k,j)+(1/2)*(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*dq(i);
        end
    end
end
C=simple(C);

ActLoc=[q2 q3];  % assume that q2 and q3 are actuated
B=transpose(jacobian(ActLoc,q));

Fx = [dq; inv(D)*(-C*dq-G)]; % model in state space form. Can only compute it explicityl for low dimensional models
Gx = [zeros(3,2); inv(D)*B];

E=0*B;

%% a few quantites for the animation of the walking motion
pH= p2;                   %hips 
pSwingLegEnd= pH + r.*[cos(q1+q2) sin(q1+q2)]; %far end of swing leg
pT=pH + [L*cos(q1+q3), L*sin(q1+q3)];  %far end of torso

vH = jacobian(pH,q)*dq;


disp('End symbolic computation of dynamics for walking model');

fcn_name='dynamic_model_3dof'; generate_model  % automatically generate m-file for ODE45 simulation

fcn_name='cartesian_pos_vel'; generate_cartesian_pos_vel  % automatically generate m-file used in animation file

save ..\functions_auto_gen\work_symbolic_model_3DOFdynamics

% ==========================
% symbolic control equations
% ==========================

% The control law design is based on the paper: E.R. Westervelt, J.W. Grizzle, and D.E. Koditschek, 
% "Hybrid Zero Dynamics of Planar Biped Walkers," IEEE-TAC, Vol. 48, No. 1, January 2003, pp. 42-56
%
disp('Begin symbolic computation of control equations for walking model');

syms s theta theta_plus theta_minus real
syms a0_q2 a1_q2 a2_q2 a3_q2 a4_q2
syms a0_q3 a1_q3 a2_q3 a3_q3 a4_q3

q2_d = bezier([a0_q2 a1_q2 a2_q2 a3_q2 a4_q2], 4, s);
q3_d = bezier([a0_q3 a1_q3 a2_q3 a3_q3 a4_q3], 4, s);
s = (theta - theta_plus)/(theta_minus - theta_plus);

theta = -q1;  % sign is chosen so that theta increases as the robot advances. This is optional.

eval(multi_eval('q2_d'));
eval(multi_eval('q3_d'));

%%% y = Hx = h_0(q) - h_d(theta(q)) 
%%% where h_0(q) specifies what is being controlled and h_d specifies its desired evolution as function of theta. 
%%% Note that h_d is based on Bezier polynmomials

Hx = simplify(eval([q2 - q2_d; q3 - q3_d]));

% Compute required derivatives for controller to impose constraint Hx == 0;
LfHx = simple(jacobian(Hx,[q dq])*Fx);
Lf2Hx = simple(jacobian(LfHx,[q dq])*Fx);
LgLfHx = simple(jacobian(LfHx,[q dq])*Gx);

disp('End symbolic computation of control equations for walking model');
fcn_name='hx_and_derivatives'; generate_output_and_derivatives  % automatically generate m-file for ODE45 simulation

save ..\functions_auto_gen\work_symbolic_outputs


if 0 %% Not used in this version of the software
    % =================================
    % symbolic modeling - zero dynamics
    % =================================
    
    disp('Begin symbolic computation of zero dynamics for walking model');
    
    
    
    sigma=jacobian(KE,dq1);
    dsigmadt=jacobian(-PE,q1);
    
    q2 = q2_d;
    q3 = q3_d;
    dq2 = jacobian(q2,q1)*dq1;
    dq3 = jacobian(q3,q1)*dq1;
    
    syms theta dtheta
    q1 = -theta;
    dq1 = -dtheta;
    eval(multi_eval('sigma'));
    eval(multi_eval('dsigmadt'));
    
    sigma = collect(sigma,dtheta);
    k1 = dtheta / sigma;
    k2 = dsigmadt;
    eval(multi_eval('k1'));
    
    disp('End symbolic computation of zero dynamics for walking model');
    save ..\functions_auto_gen\work_symbolic_zero_dynamics
    
end

% ===================================================
% symbolic modeling - 5 DOF dynamics for impact model
% ===================================================

disp('Begin symbolic computation of extended model for impact dynamics');

% For coordinate definitions, see Figure <<fig_biped_newer_coordinates.jpg>> in this directory.

syms q1 q2 q3 dq1 dq2 dq3 z1 z2 dz1 dz2 real
syms g m M Mh Mt r L  positive      

qe=[q1 q2 q3 z1 z2].';
dqe=[dq1 dq2 dq3 dz1 dz2].';

p0=[z1,z2];  % stance foot
p1=p0 + [ 1/2*r*cos(q1), 1/2*r*sin(q1)];  % center of mass of stance leg
p2=p0 + [ r*cos(q1), r*sin(q1)];   % hip
p3=p2 + [1/2*r*cos(q1+q2), 1/2*r*sin(q1+q2)]; % center of mass of swing leg
p4=p2 + [L*cos(q1+q3), L*sin(q1+q3)]; % center of mass of torso

PE=g*m*p1(2)+g*Mh*p2(2)+g*m*p3(2)+g*Mt*p4(2);
PE=simple(PE);

v1=jacobian(p1,qe)*dqe;
v2=jacobian(p2,qe)*dqe;
v3=jacobian(p3,qe)*dqe;
v4=jacobian(p4,qe)*dqe;

KE1=simplify((1/2)*m*v1.'*v1);
KE2=simplify((1/2)*Mh*v2.'*v2);
KE3=simplify((1/2)*m*v3.'*v3);
KE4=simplify((1/2)*Mt*v4.'*v4);

KE=simple(KE1+KE2+KE3+KE4);

De=simple(jacobian(KE,dqe).');
De=simple(jacobian(De,dqe));

ImpactForceLoc=[z1+r*cos(q1)+r*cos(q1+q2) z2+r*sin(q1)+r*sin(q1+q2)]; % swing leg end
E=simple(jacobian(ImpactForceLoc,qe).');

B=0*E;D=De;C=0*D;G=0*D(:,1); q=qe;dq=dqe;% Ony D and E are needed for the impact model. The other matrices are set to zero to speed m-file execution.

disp('End symbolic computation of extended model for impact dynamics');
fcn_name='dynamic_model_for_impacts'; generate_model  % automatically generate m-file for building impact model

save ..\functions_auto_gen\work_symbolic_impact