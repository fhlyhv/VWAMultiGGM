function y = denominator(x,p)

% d: dimension
% Yu Hang, Nov. 2016, NTU


y = sum(psi(1,x+(1-(1:p))/2)-1/x)/2;
if y == 0
    x = x+(1-(1:p))/2;
    y = sum(1./x-1/x(1)+1./(2*x.^2)+1./(6*x.^3)-1./(30*x.^5)+1./(42*x.^7) ...
        -1./(30*x.^9)+5./(66*x.^11)-691./(2730*x.^13)+7./(6*x.^15))/2;
end