function K = fixlength(s1,s2,len,indent)
%FIXLENGTH Returns a string which has been divided up into < LEN
%character chuncks with the '...' line divide appended at the end
%of each chunck.
%   FIXLENGTH(S1,S2,L) is string S1 with string S2 used as break points into
%   chuncks less than length L.

%Eric Westervelt
%5/31/00
%1/11/01 - updated to use more than one dividing string
%4/29/01 - updated to allow for an indent

tmp=s1;
K=[];
count=0;
while length(tmp) > len,
  I = [];  
  for c = 1:length(s2)
    I = [I findstr(tmp,s2(c))];
    I = sort(I);
  end
  if isempty(I) & count == 0
    K = [];
    error('S2 does not exist in S1')
  end
  II = find(I <= len);
  if isempty(II)
    K = [];
    error('Cannot fixlength of S1 on basis of S2')
  end
  if nargin > 3
    K = [K,tmp(1:I(II(length(II)))),' ...',10,indent];
  else
    K = [K,tmp(1:I(II(length(II)))),' ...',10];
  end
  tmp = tmp(I(II(length(II)))+1:length(tmp));
  count = count+1;
end
K = [K,tmp];