function [exec_string]=multi_eval(var_string)

  % a brute-force function used to enumuate a symbolic expression
  % over and over until all subvariables have been substituted
  % usage: u = eval(multi_eval(u));

  % multi_eval() is called by 'symb_model_biped_3dof.m'
  
  
  code_string = '; exit_loop = 0; while(not(exit_loop)) tem2 = eval(tem1); str_tem2 = char(tem2); exit_loop = strcmp(char(tem2),char(tem1)); tem1 = tem2; ';
  exec_string = [ 'tem1 = ' var_string code_string var_string '= tem2; end']; 