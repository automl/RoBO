syms theta1 theta2 theta3 dtheta1 dtheta2 dtheta3  real
syms g m M Mh Mt r l  positive

theta=[theta1 theta2 theta3].';
dtheta=[dtheta1 dtheta2 dtheta3].';

p1=(r/2)*[sin(theta1),cos(theta1)];  % center of gravity of stance leg
p2=2*p1;   % hip
p3=p2+(r/2)*[sin(-theta2),-cos(-theta2)];  % center of gravity of swing leg
p4=p2+l*[sin(theta3),cos(theta3)];  % center of gravity of torso

PE=g*m*p1(2)+g*Mh*p2(2)+g*m*p3(2)+g*Mt*p4(2);
PE=simple(PE);

v1=jacobian(p1,theta)*dtheta;
v2=jacobian(p2,theta)*dtheta;
v3=jacobian(p3,theta)*dtheta;
v4=jacobian(p4,theta)*dtheta;

KE1=simplify((1/2)*m*v1.'*v1);
KE2=simplify((1/2)*Mh*v2.'*v2);
KE3=simplify((1/2)*m*v3.'*v3);
KE4=simplify((1/2)*Mt*v4.'*v4);

KE=simple(KE1+KE2+KE3+KE4);

KE_old=KE;

L=KE-PE;

G=jacobian(PE,theta).';
G=simple(G);
D=simple(jacobian(KE,dtheta).');
D=simple(jacobian(D,dtheta));

syms C real
n=max(size(theta));
for k=1:n
   for j=1:n
      C(k,j)=0*g;
      for i=1:n
         C(k,j)=C(k,j)+(1/2)*(diff(D(k,j),theta(i))+diff(D(k,i),theta(j))-diff(D(i,j),theta(k)))*dtheta(i);
      end
   end
end
C=simple(C);

return

