%
% File to automatically build up the .m-files needed for our simualtor
%

% generate_model

disp(['[creating ',upper(fcn_name),'.m]']);
fid = fopen(['..\functions_auto_gen\',fcn_name,'.m'],'w');
n=max(size(q));
fprintf(fid,['function [D,C,G,B,E]='...
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
fprintf(fid,'\n%s','%% Model NOTATION: Spong and Vidyasagar, page 142, Eq. (6.3.12)');
fprintf(fid,'\n%s','%%                 D(q)ddq + C(q,dq)*dq + G(q) = B*Motor_Torques + E*External_Forces');
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
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',['D=zeros(',num2str(n),',',num2str(n),');']);
for i=1:n
    for j=1:n
        Temp0=D(i,j);
        if Temp0 ~= 0
            Temp1=char(Temp0);
            Temp2=['D(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
        end
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',['C=zeros(',num2str(n),',',num2str(n),');']);
for i=1:n
    for j=1:n
        Temp0=C(i,j);
        if Temp0 ~= 0
            %ttt = char(vectorize(jac_P(2)));
            Temp1=char(Temp0);
            Temp2=['C(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
        end
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s',['G=zeros(',num2str(n),',1);']);
for i=1:n
    Temp0=G(i);
    if Temp0 ~= 0
        Temp1=char(Temp0);
        Temp2=['G(',num2str(i),')=',Temp1,';'];
        Temp3=fixlength(Temp2,'*+-',65,'         ');
        fprintf(fid,'\n%s',Temp3);
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
[n,m]=size(B);
fprintf(fid,'\n%s',['B=zeros(',num2str(n),',',num2str(m),');']);
for i=1:n
    for j=1:m
        Temp0=B(i,j);
        if Temp0 ~= 0
            Temp1=char(Temp0);
            Temp2=['B(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
        end
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
[n,m]=size(E);
fprintf(fid,'\n%s',['E=zeros(',num2str(n),',',num2str(m),');']);
for i=1:n
    for j=1:m
        Temp0=E(i,j);
        if Temp0 ~= 0
            Temp1=char(Temp0);
            Temp2=['E(',num2str(i),',',num2str(j),')=',Temp1,';'];
            Temp3=fixlength(Temp2,'*+-',65,'         ');
            fprintf(fid,'\n%s',Temp3);
        end
    end
end
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','%%');
fprintf(fid,'\n%s','return');
status = fclose(fid)


return