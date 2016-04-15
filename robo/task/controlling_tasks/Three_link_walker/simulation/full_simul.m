function [x,t]=full_simul(x0,a,theta_minus,steps,do_animation,draw_graphs)

% it's not necessary to pass theta
theta_plus = delta_theta(a,theta_minus);

% 'tfinal' is the amount of time the system is given to experience an impact.
% If an impact does not happen within this amount of time, ODE45aborts.
tstart = 0;
tfinal = 5; 

% Choice of feedback control method
FT=1; % 1 = Bhat_Bernstein finite-time controller and ~1 = a high-gain linear controller

options = odeset('Events',@impulse_event);

x_concat = [];
t_concat = [];
swap_times = [];
tmax = 0;

disp('====================');

i=0;
while (i < steps)

    i = i+1;
    [t,x,te,xe] = ode45(@f,[tstart tfinal],x0,options,a,theta_minus,theta_plus);
    
    % concatenate the results of this step to results of previous step
    
    nt = length(t);
    xout = [x(2:nt,:)];
    tout = [t(2:nt)];
    
    x_concat = [x_concat; xout];
    t_concat = [t_concat; tout+tmax];
    tmax = max(t_concat);
    
    %if a step did not occur, stop simulating without an error
    
    if (isempty(te))
        i = steps+1; % this will cause the 'while' loop to exit.
    else
        [x0,Ft,Fn] = delta(xe);
        disp(sprintf('Step %i: \t mu=%f\n',i,abs(Ft/Fn)));
        swap_times = [swap_times tmax];
    end
end

x = x_concat;
t = t_concat;

if (draw_graphs == 1)
    
    % Calculate how well the outputs have been driven to zero,
    % which determines how well the zero dynamcis are being imposed. 
    
    for i = 1:1:size(x_concat,1) 
        q=x(i,1:3); dq=x(i,4:6);
        temp = hx_and_derivatives(q,dq,a,theta_minus,theta_plus);
        Hx(i,:)=temp.';
    end
    
    figure;
    plot(t_concat,Hx(:,1));
    title('\fontsize{12}\bfError of swing leg tracking (rad)');
    xlabel('\fontsize{10}time');
    ylabel('\fontsize{10}q^{d}_2-q_2');
    
    figure;
    plot(t_concat,Hx(:,2));
    title('\fontsize{12}\bfError of torso tracking (rad)');
    xlabel('\fontsize{10}time');
    ylabel('\fontsize{10}q^{d}_3-q_3');
    
    % Plot relative joint angles and velocities
 
    q=x(:,1:3);
    dq=x(:,4:6);

    figure;
    plot(t_concat,q)
    title('\fontsize{12}\bfRelative joint angles (rad)');
    legend('q_1','q_2','q_3');
    xlabel('\fontsize{10}time');
    ylabel('\fontsize{10}q_i');
    
    figure;
    plot(t_concat,dq)
    title('\fontsize{12}\bfRelative joint velocities (rad/s)');
    legend('dq_1/dt','dq_2/dt','dq_3/dt');
    xlabel('\fontsize{10}time');
    ylabel('\fontsize{10}dq_i/dt');
    
    % Calculate control effort (not recorded as a state in simulation, so we have to re-generate it)
    
    for k=1:1:size(x,1)
        
      q=x(k,1:3); dq=x(k,4:6);
      [Hx,LfHx,Lf2Hx,LgLfHx]= hx_and_derivatives(q,dq,a,theta_minus,theta_plus);
    
      y = Hx;
      dy = LfHx;

      if (1) % 1 = use Bhat-Bernstein Finite-time converging controller  
        alpha=.9;
        epsilon=0.05;
        p=length(y);    
        nu=zeros(size(y));
        for i=1:p
          nu(i)= BB(y(i),dy(i),alpha,epsilon);
        end
          tem=inv(LgLfHx)*(nu-Lf2Hx);    
          u(k,1)=tem(1);
          u(k,2)=tem(2);
        else % use high-gain I-O linearizing controller 
          tem=inv(LgLfHx)*(-Lf2Hx-10000*y-300*dy);
          u(k,1)=tem(1);
          u(k,2)=tem(2);
        end
    end

    figure;
    plot(t_concat,u(:,1),'r');
    hold on
    plot(t_concat,u(:,2),'b');
    title('\fontsize{12}\bfJoint control effort (Nm)');
    legend('u_1 (between q_1 and q_2)','u_2 (between q_1 and q_3)');
    xlabel('\fontsize{10}time');
    ylabel('\fontsize{10}control effort: u_i');
  
end

if (do_animation == 1)
    anim(x_concat,t_concat,swap_times);
end

%-------------------------------------------------------------------

function dx = f(t,x,a,theta_minus,theta_plus)

q=x(1:3); dq=x(4:6);

% description of instantaneous system dynamics
[D,C,G,B,E] = dynamic_model_3dof(q,dq);
[Hx,LfHx,Lf2Hx,LgLfHx]= hx_and_derivatives(q,dq,a,theta_minus,theta_plus);

% compute vector fields for model here
Fx = [x(4:6); inv(D)*(-C*x(4:6)-G)];
Gx = [zeros(3,2); inv(D)*B];

% implement control here
y = Hx;
dy = LfHx;

if (1) % 1 = use Bhat-Bernstein Finite-time converging controller  
    alpha=.9;
    epsilon=0.05;
    p=length(y);    
    nu=zeros(size(y));
    for i=1:p
        nu(i)= BB(y(i),dy(i),alpha,epsilon);
    end
    u=inv(LgLfHx)*(nu-Lf2Hx);    
else % use high-gain I-O linearizing controller 
    u=inv(LgLfHx)*(-Lf2Hx-10000*y-300*dy);
end

%% state space takes the form [q1 q2 q3 dq1 dq2 dq3]'
dx = Fx+Gx*u; 

function [nu,V]=BB(y,dy,alpha,epsilon)  % Computes Bhat-Bernstein 
%                                         Finite-Time Controller for a Double Integrator

sigma=1/2;
rho=2;

dy=epsilon*dy;
phi=y+1/(2-alpha)*sign(dy)*abs(dy)^(2-alpha);
nu=(-sign(dy)*abs(dy)^alpha-sign(phi)*abs(phi)^(alpha/(2-alpha)))/epsilon^2;
V=(2-alpha)/(3-alpha)*abs(phi)^((3-alpha)/(2-alpha)) + sigma*dy*phi ...;
    +rho/(3-alpha)*abs(dy)^(3-alpha);
if V < -1
    nu=0;
end

return

%-------------------------------------------------------------------

function [value,isterminal,direction] = impulse_event(t,x,a,theta_minus,theta_plus)
% Locate the time when the end of the swing leg passes through zero

value = q_to_theta(x(1:3)) - theta_minus;      
isterminal = 1;                                
direction = 0;                                 

%-------------------------------------------------------------------

function [x_plus_afterswap,Ft,Fn] = delta(x_minus)

% calculate instantaneous impact dynamics
q=x_minus(1:3); dq=x_minus(4:6);
[De,Ce,Ge,Be,E]= dynamic_model_for_impacts(q,dq);

dxe_minus=[x_minus(4) x_minus(5) x_minus(6) 0 0];

A=[De -E; E' zeros(2)];
b=[De*dxe_minus'; 0; 0];
c=inv(A)*b;

% state after impact, but before leg swap (only velocities change)
x_plus_beforeswap=[x_minus(1) x_minus(2) x_minus(3) c(1) c(2) c(3)]; 

% perform leg swap
tem=[1 1 0; 0 -1 0; 0 -1 1];
S=[tem zeros(3,3); zeros(3,3) tem];
x_plus_afterswap=x_plus_beforeswap*S' + [pi -2*pi -pi 0 0 0];

Ft=c(6);
Fn=c(7);  

if (Fn < 0) disp('Illegal impact condition: normal force pulling robot downward - violates unilateral constraint'); end
if (abs(Ft/Fn) > 0.8) disp ('Illegal impact condition: required coefficient of friction too high'); end

%-------------------------------------------------------------------
function anim(x,t,swap_times)

[r,m,Mh,Mt,L,g]=model_params;
fig1=figure;
set(fig1,'Position', [120 65 400 200]);

timescale=10;
horiz_reference=0;

for i=1:1:size(x,1)-1
    
    [pT,pSwingLegEnd,pH]= cartesian_pos_vel(x(i,1:3),x(i,4:6)); 
    
    pt0=[0 0];          %pivot  
    pt1=pH.';           %hips 
    pt2=pSwingLegEnd.'; %far end of advancing leg
    pt3=pT.';           %end of torso
    
    if (i==1) horiz_reference=-pt2(1); end
    if (i==1) hips_reference=pt1(1); end
   
    shift_vec=[pt1(1) 0]; %use this to center the x-position of the hips in the viewscreen
    
    pt0=pt0-shift_vec;
    pt1=pt1-shift_vec;
    pt2=pt2-shift_vec;
    pt3=pt3-shift_vec;
    
    clf;
    axis([-2 2 -1 2.0]);
    axis off;
       
    pt_line(pt0, pt1,[ 0.000 0.690 0.000 ],2);
    pt_circle((pt0+pt1)./2,0.05,[ 0.000 0.690 0.000 ]);
        
    pt_line(pt1, pt2,'r',2);
    pt_circle((pt1+pt2)./2,0.05,'r');
        
    pt_line(pt1, pt3,'b',2);
    pt_circle(pt3,0.05,'b');
    pt_circle(pt1,0.05,'b');
      
    pt_line([-2.2 0], [2.2 0],'k',2);
        
    for j=floor(horiz_reference)-1:1:floor(horiz_reference)+2
        text(-shift_vec(1)-horiz_reference+j,-0.2,num2str(j));
    end
        
    text(-1.5,-0.7,['time: ',num2str(floor(t(i)*100)/100)]);
    text(-1.5,-1.1,['hips-dist: ',num2str(floor(((horiz_reference+shift_vec(1))*100))/100)]);
    text(0.5,-0.7,['avg-hip-veloc: ',num2str(floor((10*horiz_reference+shift_vec(1)-hips_reference)/t(i))/10)]);
    
    drawnow; 
    
    if sum(swap_times==t(i))
      horiz_reference=horiz_reference+pt2(1)+shift_vec(1);
    end
    
    delay((t(i+1)-t(i))/timescale);
end


