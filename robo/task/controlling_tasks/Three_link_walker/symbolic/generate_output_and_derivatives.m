%
%
% File to automatically build up some of the .m-files needed for the simualtor
%
% generate_output_and_derivatives


disp(['[creating ',upper(fcn_name),'.m]']);
fid = fopen(['..\functions_auto_gen\',fcn_name,'.m'],'w');
fprintf(fid,['function [Hx,LfHx,Lf2Hx,LgLfHx]=' ...
        ' %s(q,dq,a,theta_minus,theta_plus)'],fcn_name);
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',['%%  ',upper(fcn_name)]);
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',['%%  ',datestr(now)]);
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%% Author(s): Ben Morris and Jessy Grizzle');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','[r,m,Mh,Mt,L,g]=model_params;'); % HERE put your file for model parameters
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','q1=q(1);q2=q(2);q3=q(3);'); % HERE adjust for number of dof
fprintf(fid,'\n%s','dq1=dq(1);dq2=dq(2);dq3=dq(3);');    
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','a0_q2 = a(1,1); a1_q2 = a(1,2); a2_q2 = a(1,3); a3_q2 = a(1,4); a4_q2 = a(1,5);'); % HERE adjust for number coefficients
fprintf(fid,'\n%s','a0_q3 = a(2,1); a1_q3 = a(2,2); a2_q3 = a(2,3); a3_q3 = a(2,4); a4_q3 = a(2,5);');    
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',' ');
[n,m]=size(Hx);
for i=1:n
    for j=1:m
        Temp0=Hx(i,j);
            Temp1=char(Temp0);
            Temp2=['Hx(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
    end
end

fprintf(fid,'\n%s',' ');
[n,m]=size(LfHx);
for i=1:n
    for j=1:m
        Temp0=LfHx(i,j);
            Temp1=char(Temp0);
            Temp2=['LfHx(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
    end
end

fprintf(fid,'\n%s',' ');
[n,m]=size(Lf2Hx);
for i=1:n
    for j=1:m
        Temp0=Lf2Hx(i,j);
            Temp1=char(Temp0);
            Temp2=['Lf2Hx(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
    end
end

fprintf(fid,'\n%s',' ');
[n,m]=size(LgLfHx);
for i=1:n
    for j=1:m
        Temp0=LgLfHx(i,j);
            Temp1=char(Temp0);
            Temp2=['LgLfHx(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
    end
end
fprintf(fid,'\n%s','return');
status = fclose(fid)

return