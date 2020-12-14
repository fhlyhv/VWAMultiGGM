function ELBO = ELBO_overall_multi_oneomega(a1,b1,V,nu,ET,EK,EKsum2,EKinv,ELAMBDA,expd,nS,idm,nscale,nnodes,n,a,b,d,KCF)

% ELAMBDA needs to be updated before inputing into the function

ELBO = (n/2-a1)*psi(a1)-n/2*log(b1)-EK{1}*nS(1)/2+a1+gammaln(a1)-logdet(KCF)/2*n+a+gammaln(a)-a*psi(a);

for m = 2:nscale
    ELBO = ELBO +(nnodes(m)+1+n)/2*logdet(V{m})+logMvGamma(nu(m)/2,nnodes(m))+(nnodes(m)+1+n-nu(m))/2*MvPolyGamma(nu(m)/2,nnodes(m),0)+nu(m)*nnodes(m)/2 ...
        -sum(sum(EK{m}.*nS(idm{m},idm{m})))/2-sum(sum(ET(idm{m},idm{m-1}).*nS(idm{m},idm{m-1}))) ...
        -sum(sum((ET(idm{m-1},idm{m})*EKinv{m}*ET(idm{m},idm{m-1})).*nS(idm{m-1},idm{m-1})))/2 ...
        -sum(sum(ELAMBDA{m}.*EKsum2{m}))/4+sum(log(expd{m}))-sum(d{m})+sum(1./expd{m}) ...  %sum(log(expint(d{m})))
        +nnodes(m)*(nnodes(m)-1)/4*psi(a)-nnodes(m)*(nnodes(m)-1)/4*log(b);
end
    
    
    






