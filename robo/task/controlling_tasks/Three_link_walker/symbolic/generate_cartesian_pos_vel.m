%
% File to automatically build up the .m-files needed for our simualtor
%

% generate_cartesian_pos_vel

disp(['[creating ',upper(fcn_name),'.m]']);
fid = fopen(['..\functions_auto_gen\',fcn_name,'.m'],'w');
n=max(size(q));
fprintf(fid,['function [pT,pSwingLegEnd,pH,vH]='...
        ' %s(q,dq)'],fcn_name);
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
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','q1=q(1);q2=q(2);q3=q(3);'); % HERE modify for your number of variables  
fprintf(fid,'\n%s','dq1=dq(1);dq2=dq(2);dq3=dq(3);'); 
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
n=2;
fprintf(fid,'\n%s',['pH=zeros(',num2str(n),',1);']);
for i=1:n
    Temp0=pH(i);
    if Temp0 ~= 0
        Temp1=char(Temp0);
        Temp2=['pH(',num2str(i),')=',Temp1,';'];
        Temp3=fixlength(Temp2,'*+-',65,'         ');
        fprintf(fid,'\n%s',Temp3);
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',['pT=zeros(',num2str(n),',1);']);
for i=1:n
    Temp0=pT(i);
    if Temp0 ~= 0
        Temp1=char(Temp0);
        Temp2=['pT(',num2str(i),')=',Temp1,';'];
        Temp3=fixlength(Temp2,'*+-',65,'         ');
        fprintf(fid,'\n%s',Temp3);
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',['pSwingLegEnd=zeros(',num2str(n),',1);']);
for i=1:n
    Temp0=pSwingLegEnd(i);
    if Temp0 ~= 0
        Temp1=char(Temp0);
        Temp2=['pSwingLegEnd(',num2str(i),')=',Temp1,';'];
        Temp3=fixlength(Temp2,'*+-',65,'         ');
        fprintf(fid,'\n%s',Temp3);
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',['vH=zeros(',num2str(n),',1);']);
for i=1:n
    Temp0=vH(i);
    if Temp0 ~= 0
        Temp1=char(Temp0);
        Temp2=['vH(',num2str(i),')=',Temp1,';'];
        Temp3=fixlength(Temp2,'*+-',65,'         ');
        fprintf(fid,'\n%s',Temp3);
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','return');
status = fclose(fid)
return