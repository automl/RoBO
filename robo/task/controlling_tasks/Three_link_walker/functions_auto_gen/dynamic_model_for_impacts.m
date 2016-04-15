function [D,C,G,B,E]= dynamic_model_for_impacts(q,dq)
%%
%%
%%  DYNAMIC_MODEL_FOR_IMPACTS
%%
%%
%%  03-Dec-2003 17:18:28
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
D=zeros(5,5);
D(1,1)=(Mt+3/2*m+Mh+m*cos(q2))*r^2+2*Mt*L*cos(q3)*r+Mt*L^2;
D(1,2)=(1/4*m+1/2*m*cos(q2))*r^2;
D(1,3)=Mt*L*cos(q3)*r+Mt*L^2;
D(1,4)=(-Mh*sin(q1)-Mt*sin(q1)-1/2*m*sin(q1+q2)-3/2*m*sin(q1))*r- ...
         Mt*L*sin(q1+q3);
D(1,5)=(3/2*m*cos(q1)+1/2*cos(q1+q2)*m+Mt*cos(q1)+Mh*cos(q1))*r+ ...
         Mt*L*cos(q1+q3);
D(2,1)=(1/4*m+1/2*m*cos(q2))*r^2;
D(2,2)=1/4*m*r^2;
D(2,4)=-1/2*m*sin(q1+q2)*r;
D(2,5)=1/2*cos(q1+q2)*m*r;
D(3,1)=Mt*L*cos(q3)*r+Mt*L^2;
D(3,3)=Mt*L^2;
D(3,4)=-Mt*L*sin(q1+q3);
D(3,5)=Mt*L*cos(q1+q3);
D(4,1)=(-Mh*sin(q1)-Mt*sin(q1)-1/2*m*sin(q1+q2)-3/2*m*sin(q1))*r- ...
         Mt*L*sin(q1+q3);
D(4,2)=-1/2*m*sin(q1+q2)*r;
D(4,3)=-Mt*L*sin(q1+q3);
D(4,4)=2*m+Mt+Mh;
D(5,1)=(3/2*m*cos(q1)+1/2*cos(q1+q2)*m+Mt*cos(q1)+Mh*cos(q1))*r+ ...
         Mt*L*cos(q1+q3);
D(5,2)=1/2*cos(q1+q2)*m*r;
D(5,3)=Mt*L*cos(q1+q3);
D(5,5)=2*m+Mt+Mh;
%%
%%
%%
%%
C=zeros(5,5);
%%
%%
%%
%%
G=zeros(5,1);
%%
%%
%%
%%
B=zeros(5,2);
%%
%%
%%
%%
E=zeros(5,2);
E(1,1)=-r*sin(q1)-r*sin(q1+q2);
E(1,2)=cos(q1)*r+r*cos(q1+q2);
E(2,1)=-r*sin(q1+q2);
E(2,2)=r*cos(q1+q2);
E(4,1)=1;
E(5,2)=1;
%%
%%
return