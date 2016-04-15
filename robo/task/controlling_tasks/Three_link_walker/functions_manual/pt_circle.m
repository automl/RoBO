function [obj]=pt_circle(pt0,rad,color)
  
  %draws a circle around a certain point
  %pt_circle() is called by 'full_simul.m'

  i=0:0.1:2*pi;
  x=rad.*cos(i)+pt0(1);
  y=rad.*sin(i)+pt0(2);
  obj=patch(x,y,color);
  set(obj,'EdgeColor','none');
  