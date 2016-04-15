function [D,C,G,B,E]= dynamic_model_3dof(q,dq)
%%
%%
%%  DYNAMIC_MODEL_3DOF
%%
%%
%%  03-Dec-2003 17:17:57
%%
%%
%% Author(s): Ben Morris and Jessy Grizzle
%%
%%
%% Model NOTATION: Spong and Vidyasagar, page 142, Eq. (6.3.12)
%%                 D(q)ddq + C(q,dq)*dq + G(q) = B*Motor_Torques + E*External_Forces
%%
%%
[r,m,Mh,Mt,L,g]=model_params;
%%
%%
%%
%%
q1=q(1);q2=q(2);q3=q(3);
dq1=dq(1);dq2=dq(2);dq3=dq(3);
%%
%%
%%
%%
D=zeros(3,3);
D(1,1)=(m*cos(q2)+Mt+3/2*m+Mh)*r^2+2*Mt*r*L*cos(q3)+Mt*L^2;
D(1,2)=(1/2*m*cos(q2)+1/4*m)*r^2;
D(1,3)=Mt*r*L*cos(q3)+Mt*L^2;
D(2,1)=1/4*m*r^2*(2*cos(q2)+1);
D(2,2)=1/4*m*r^2;
D(3,1)=Mt*r*L*cos(q3)+Mt*L^2;
D(3,3)=Mt*L^2;
%%
%%
%%
%%
C=zeros(3,3);
C(1,1)=-1/2*r*(m*sin(q2)*r*dq2+2*Mt*L*sin(q3)*dq3);
C(1,2)=-1/2*m*sin(q2)*r^2*(dq1+dq2);
C(1,3)=-Mt*r*L*sin(q3)*(dq1+dq3);
C(2,1)=1/2*m*sin(q2)*r^2*dq1;
C(3,1)=Mt*r*L*sin(q3)*dq1;
%%
%%
%%
%%
G=zeros(3,1);
G(1)=1/2*g*(2*Mh*cos(q1)+m*cos(q1+q2)+2*Mt*cos(q1)+3*m*cos(q1))* ...
         r+g*Mt*L*cos(q1+q3);
G(2)=1/2*g*m*r*cos(q1+q2);
G(3)=g*Mt*L*cos(q1+q3);
%%
%%
%%
%%
B=zeros(3,2);
B(2,1)=1;
B(3,2)=1;
%%
%%
%%
%%
E=zeros(3,2);
%%
%%
return