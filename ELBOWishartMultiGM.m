function ELBO = ELBOWishartMultiGM(nu,Vinv,P,m,nscale,n,tmp5,Am,Tcoaser,ELAMBDAm,ELAMBDAmm1)
[~,q] = chol(Vinv);

if q ~= 0 || nu <= P+3
    ELBO = -Inf;
else
    nump = nu-P;
    V = inv(Vinv);
    V = (V+V')/2;
    logdetV = logdet(V); %-logdet(Vinv);
    psipnuo2 = MvPolyGamma(nu/2,P,0);
    HpnElogdetK = (P+1+n)/2*logdetV+logMvGamma(nu/2,P)-(nump-n-1)/2*psipnuo2+nu*P/2;
    EK = nu*V;
    VK = nu*(V.*V+diag(V)*diag(V)');
    TcVinvTc = Tcoaser'*Vinv*Tcoaser;
    if m == 2
        ELBO = -sum(sum(tmp5.*EK))/2-sum(sum(TcVinvTc.*Am))/(nump-1)/2 ...
            -sum(sum(ELAMBDAm.*(EK.^2+VK)))/4+HpnElogdetK;
    elseif m == nscale
        ELBO = -sum(sum(tmp5.*EK))/2-sum(sum(TcVinvTc.*Am))/(nump-1)/2 ...
            -sum(sum(ELAMBDAm.*(EK.^2+VK)))/4 ...
            -(sum(sum(ELAMBDAmm1.*TcVinvTc.^2))+diag(TcVinvTc)'*ELAMBDAmm1*diag(TcVinvTc)/(nump-1))/4/nump/(nump-3)+HpnElogdetK;
    else
        ELBO = -sum(sum(tmp5.*EK))/2-sum(sum(TcVinvTc.*Am))/(nump-1)/2 ...
            -sum(sum(ELAMBDAm.*(EK.^2+VK)))/4 ...
            -(sum(sum(ELAMBDAmm1.*TcVinvTc.^2))+diag(TcVinvTc)'*ELAMBDAmm1*diag(TcVinvTc)/(nump-1))/4/nump/(nump-3)+HpnElogdetK;
    end

end





