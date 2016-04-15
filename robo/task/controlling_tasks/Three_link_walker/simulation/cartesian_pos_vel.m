function [pT,pSwingLegEnd,pH,vH]= cartesian_pos_vel(q,dq)
%%
%%
%%  CARTESIAN_POS_VEL
%%
%%
%%  03-Dec-2003 17:17:57
%%
%%
%% Author(s): Ben Morris and Jessy Grizzle
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
pH=zeros(2,1);
pH(1)=r*cos(q1);
pH(2)=r*sin(q1);
%%
%%
pT=zeros(2,1);
pT(1)=r*cos(q1)+L*cos(q1+q3);
pT(2)=r*sin(q1)+L*sin(q1+q3);
%%
%%
pSwingLegEnd=zeros(2,1);
pSwingLegEnd(1)=r*cos(q1)+r*cos(q1+q2);
pSwingLegEnd(2)=r*sin(q1)+r*sin(q1+q2);
%%
%%
vH=zeros(2,1);
vH(1)=-r*sin(q1)*dq1;
vH(2)=r*cos(q1)*dq1;
%%
%%
return