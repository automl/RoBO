function [obj]=pt_line(pt0,pt1,color,thickness)

  %draws a line between two points
  %pt_line() is called by 'full_simul.m'

  obj=line([pt0(1) pt1(1)],[pt0(2) pt1(2)]);
  set(obj,'Color',color);
  set(obj,'LineWidth', thickness);