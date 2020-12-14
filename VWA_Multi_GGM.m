function [V,nu,EKall,Kest] = VWA_Multi_GGM(data, n_nodes, Tree)

% variational Wishart approximation using natural gradients
% Inputs:
%       data -- n x p matrix with n samples and p vairables in the finest
%               scale
%       n_nodes -- number of nodes in each scale, from coarsest to finest
%       Tree -- the adjacency matrix of the latent tree whose finest scale
%               is observed
% Outputs:
%       V, nu -- estimated scale matrix and degree of freedom of the
%                Wishart distribution in each scale
%       EKall -- expectation of the overall joint precision matrix
%       Kest  -- estimated overall precision matrix after thresholding the
%                estimating the nonzero elements
% Yu Hang, Nov, 2018, NTU

eta0 = 10; %5; %0.05; %0.01; %1e-3; %
etaT0 = 1e-2;
tolr = 1e-4; %max(1e-4,min(eta0/100,1e-2));
tolm = 2e-2; %max(1e-3,min(eta0/100,1e-2));
tolg = 1e-3;
maxIter = 1e3;
tic;
%% initialization
[n,pf] = size(data);
p = size(Tree,1);
pc = p - pf;

nSf = data'*data;
nS = zeros(p);
nS(pc+1:end,pc+1:end) = nSf;

offset = [0; cumsum(n_nodes)];
nscale = length(n_nodes);

Tree = spdiags(zeros(p),0,-abs(sign(Tree)));
Tree = spdiags(-sum(Tree,2)+0.1,0,Tree);
[idr_T,idc_T] = find(tril(Tree,-1));
idl_T = idr_T + (idc_T-1)*p;

% ET = full(T);
% load polyET; %ET_fBM; %
ET = gaussEM(Tree,nSf/n,1,1e-2);

nu = n_nodes+n+1;
V = cell(nscale,1);
Vinv = cell(nscale,1);
plbd = NaN*ones(nscale,1);
idl_lbd = cell(nscale,1);
idu_lbd = cell(nscale,1);
d = cell(nscale,1);
ELAMBDA = cell(nscale,1);
EK = cell(nscale,1);
EK0 = cell(nscale,1);
EKinv = cell(nscale,1);
TVinvT = cell(nscale,1);
gradEK = cell(nscale,1);
gradElogdetK = NaN*ones(nscale,1);
VARKdnu = cell(nscale,1);
EK2 = cell(nscale,1);
gradET = NaN*ones(p);
EKsum2 = cell(nscale,1);
Elambda = cell(nscale,1);
idm = cell(nscale,1);
expd = cell(nscale,1);
Am = cell(nscale,1);


for m =  nscale:-1:1
    idm{m} = offset(m)+1:offset(m+1);
    if m == nscale
        EK0{m} = ET(pc+1:end,pc+1:end);
    else
        TVinvT{m} = ET(idm{m},idm{m+1})*Vinv{m+1}*ET(idm{m+1},idm{m});
        EK0{m} = ET(idm{m},idm{m})-ET(idm{m},idm{m+1})/EK0{m+1}*ET(idm{m+1},idm{m});
    end
    EK0{m} = (EK0{m}+EK0{m}')/2;
    V{m} = EK0{m}/nu(m);
    Vinv{m} = inv(V{m}); 
    EK{m} = nu(m)*V{m};
    VARKdnu{m} = V{m}.^2+diag(V{m})*diag(V{m})';
    EKinv{m} = Vinv{m}/(nu(m)-n_nodes(m)-1);
    plbd(m) = n_nodes(m)*(n_nodes(m)-1)/2;
    [idr_lbd,idc_lbd] = find(tril(ones(n_nodes(m)),-1));
    idl_lbd{m} = idr_lbd + (idc_lbd-1)*n_nodes(m);
    idu_lbd{m} = idc_lbd + (idr_lbd-1)*n_nodes(m);
    d{m} = 0.1*ones(plbd(m),1);
    Elambda{m} = zeros(plbd(m),1);
    expd{m} = zeros(plbd(m),1);
    ELAMBDA{m} = zeros(n_nodes(m));
end



nump = nu - n_nodes;

a = sum(plbd)/2;
b = a/100;

a1 = n/2;


d1Vinv = cell(nscale,1);
d1d = cell(nscale,1);
for i = 1:nscale
    d1Vinv{i} = 0;
    d1d{i} = 0;
end
d1nu = zeros(nscale,1);

% d1b = zeros(nscale,1);
d1a1 = 0;
d1b1 = 0;
d2all = 0;

tau = 100;

eta5 = [];

KCF = ET(1:pc,1:pc);
hCF = -data*ET(pc+1:end,1:pc);
XC = hCF/KCF;
nS(1:pc,1:pc) = XC'*XC+n*inv(KCF);
nS(1:pc,pc+1:end) = XC'*data;
nS(pc+1:end,1:pc) = nS(1:pc,pc+1:end)';

Eomega = a/b; 

gamma = cell(m,1);

for m = 2:nscale
    idd = find(d{m}>=10)';
    if ~isempty(idd)
        for i = idd
            expd{m}(i) = Lentz_Algorithm(d{m}(i));
        end
    end
    expd{m}(d{m}<10) = exp(d{m}(d{m}<10)).*expint(d{m}(d{m}<10));
    Elambda{m} = 1./(expd{m}.*d{m})-1;
    gamma{m} = Elambda{m} ./ (1+Elambda{m});
    ELAMBDA{m}(idl_lbd{m}) = Eomega*Elambda{m}; %sparse(idr_lbd{m},idc_lbd{m},Eomega(m)*Elambda{m},nnodes(m+1),nnodes(m+1));
    ELAMBDA{m}(idu_lbd{m}) = ELAMBDA{m}(idl_lbd{m});
end

Am{2} = nS(idm{1},idm{1});
for m = 3:nscale
    Am{m} = nS(idm{m-1},idm{m-1})+ELAMBDA{m-1}.*EK{m-1};
end

EKall_old = ET;
gamma_old = gamma;

% b1 = Am{2}/2;
% ELBOoverall0 = -Inf; %ELBO_overall_multi_oneomega(a1,b1,V,nu,ET,EK,EKsum2,EKinv,ELAMBDA,expd,nS,idm,nscale,nnodes,n,a,b,d,KCF);
rng('default');
eta_array1 = zeros(m-1,40);
eta_array2 = zeros(1,40);
%%
for kappa = 1:maxIter
    
    
    for m = randperm(nscale) %1:nscale
        if m == 1
            b1 = Am{m+1}/2;
            EK{m} = a1/b1;
        else
            
            if m == 2
                gradaV = Vinv{m}*ET(idm{m},idm{m-1})*Am{m}/(nump(m)-1)*ET(idm{m-1},idm{m})*Vinv{m}/2 ...
                    -ELAMBDA{m}.*EK{m}/2-diag(ELAMBDA{m}*diag(EK{m}))/2;
                tmp1 = sum(sum(Am{m}.*TVinvT{m-1}))/2/(nump(m)-1)^2;
                tmp2 = sum(sum(ELAMBDA{m}.*VARKdnu{m}))/4;
                tmp = - ((n_nodes(m)+1)/nu(m)*tmp1+tmp2)/denominator(nu(m)/2,n_nodes(m));
                gradanu = tmp1 - tmp2;
                tmp5 = nS(idm{m},idm{m})+ELAMBDA{m}.*(ET(idm{m},idm{m+1})*EKinv{m+1}*ET(idm{m+1},idm{m}));
            elseif m == nscale
                gradaV = Vinv{m}*ET(idm{m},idm{m-1})*(Am{m}/(nump(m)-1)+TVinvT{m-1}.*ELAMBDA{m-1}/nump(m)/(nump(m)-3) ...
                    +diag(ELAMBDA{m-1}*diag(TVinvT{m-1})/nump(m)/(nump(m)-1)/(nump(m)-3)))*ET(idm{m-1},idm{m})*Vinv{m}/2 ...
                    -ELAMBDA{m}.*EK{m}/2-diag(ELAMBDA{m}*diag(EK{m}))/2;
                tmp1 = sum(sum(Am{m}.*TVinvT{m-1}))/2/(nump(m)-1)^2;
                tmp2 = sum(sum(ELAMBDA{m}.*VARKdnu{m}))/4;
                tmp3 = sum(sum(ELAMBDA{m-1}.*TVinvT{m-1}.^2))/4/nump(m)/(nump(m)-3);
                tmp4 = diag(TVinvT{m-1})'*ELAMBDA{m-1}*diag(TVinvT{m-1})/4/nump(m)/(nump(m)-1)/(nump(m)-3);
                tmp = - ((n_nodes(m)+1)/nu(m)*tmp1+tmp2+((n_nodes(m)+3)/(nump(m)-3)+n_nodes(m)/nump(m))/nu(m)*tmp3 ...
                    +(n_nodes(m)/nump(m)+(n_nodes(m)+1)/(nump(m)-1)+nu(m)/(nump(m)-3))/nu(m)*tmp4)/denominator(nu(m)/2,n_nodes(m));
                gradanu = tmp1 - tmp2 + (1/(nump(m)-3)+1/nump(m))*tmp3 + (1/nump(m)+1/(nump(m)-1)+1/(nump(m)-3))*tmp4;
                tmp5 = nS(idm{m},idm{m});
            else
                gradaV = Vinv{m}*ET(idm{m},idm{m-1})*(Am{m}/(nump(m)-1)+TVinvT{m-1}.*ELAMBDA{m-1}/nump(m)/(nump(m)-3) ...
                    +diag(ELAMBDA{m-1}*diag(TVinvT{m-1})/nump(m)/(nump(m)-1)/(nump(m)-3)))*ET(idm{m-1},idm{m})*Vinv{m}/2 ...
                    -ELAMBDA{m}.*EK{m}/2-diag(ELAMBDA{m}*diag(EK{m}))/2;
                tmp1 = sum(sum(Am{m}.*TVinvT{m-1}))/2/(nump(m)-1)^2;
                tmp2 = sum(sum(ELAMBDA{m}.*VARKdnu{m}))/4;
                tmp3 = sum(sum(ELAMBDA{m-1}.*TVinvT{m-1}.^2))/4/nump(m)/(nump(m)-3);
                tmp4 = diag(TVinvT{m-1})'*ELAMBDA{m-1}*diag(TVinvT{m-1})/4/nump(m)/(nump(m)-1)/(nump(m)-3);
                tmp = - ((n_nodes(m)+1)/nu(m)*tmp1+tmp2+((n_nodes(m)+3)/(nump(m)-3)+n_nodes(m)/nump(m))/nu(m)*tmp3 ...
                    +(n_nodes(m)/nump(m)+(n_nodes(m)+1)/(nump(m)-1)+nu(m)/(nump(m)-3))/nu(m)*tmp4)/denominator(nu(m)/2,n_nodes(m));
                gradanu = tmp1 - tmp2 + (1/(nump(m)-3)+1/nump(m))*tmp3 + (1/nump(m)+1/(nump(m)-1)+1/(nump(m)-3))*tmp4;
                tmp5 = nS(idm{m},idm{m})+ELAMBDA{m}.*(ET(idm{m},idm{m+1})*EKinv{m+1}*ET(idm{m+1},idm{m}));
            end
            tmp6 = ELAMBDA{m}.*EK{m} + tmp5;
            
            gradEK{m} = -tmp6/2+(gradaV+tmp*Vinv{m})/nu(m);
            gradEK{m} = (gradEK{m}+gradEK{m}')/2;
            gradElogdetK(m) = n/2 - tmp;

            %% line search
            eta = eta0;
            
            ELBO0 = ELBOWishartMultiGM(nu(m),Vinv{m},n_nodes(m),m,nscale,n,tmp5, ...
                Am{m},ET(idm{m},idm{m-1}),ELAMBDA{m},ELAMBDA{m-1});
            natgradnum = 1+2*gradElogdetK(m) - nump(m);
            natgradVinvm = -2*gradEK{m} - Vinv{m};
            gradnum = (n-nump(m)+1)/4*MvPolyGamma(nu(m)/2,n_nodes(m),1)+n_nodes(m)/2 ...
                 - sum(sum(V{m}.*tmp6))/2 +gradanu;
            gradVm = (n+n_nodes(m)+1)/2*Vinv{m}-nu(m)/2*tmp6+gradaV;
            gradVinvm = - V{m}*gradVm*V{m};
            
            sumgrad = gradnum.*natgradnum+sum(sum(gradVinvm.*natgradVinvm)); %sum(sum(tril(gradVinvm).*tril(natgradVinvm))); %
            if sumgrad < 0
                sumgrad = 0;
            end
            while 1
                nutmp = nu(m) + eta*natgradnum;
                Vinvtmp = Vinv{m} + eta*natgradVinvm;
                ELBOtmp = ELBOWishartMultiGM(nutmp,Vinvtmp,n_nodes(m),m,nscale,n,tmp5, ...
                    Am{m},ET(idm{m},idm{m-1}),ELAMBDA{m},ELAMBDA{m-1});
                if ELBOtmp >= ELBO0 + 0.01*eta*sumgrad
                    nu(m) = nutmp;
                    Vinv{m} = (Vinvtmp+Vinvtmp')/2;
%                     if eta == eta0
%                         eta;
%                     end
                    break;
                end
%                 if eta <0.2
                    eta = eta/2;
%                 else
%                     eta = eta-0.1;
%                 end
            end
            eta_array1(m-1,kappa) = eta;
            
            V{m} = inv(Vinv{m});
            V{m} = (V{m}+V{m})/2;
            EKinv{m} = Vinv{m}/(nu(m)-n_nodes(m)-1);
            EK{m} = nu(m)*V{m};
            VARKdnu{m} = V{m}.^2+diag(V{m})*diag(V{m})';
            TVinvT{m-1} = ET(idm{m-1},idm{m})*Vinv{m}*ET(idm{m},idm{m-1});
            
            if m < nscale
                Am{m+1} = nS(idm{m},idm{m})+ELAMBDA{m}.*EK{m};
            end
        end
    end
    nump = nu - n_nodes;
    
    ELBO0 = ELBOTree(Vinv,nump,ET,EK,ELAMBDA,nS,idm,nscale);
    ET0 = ET;
    for iter = 1:100
        for m = 2:nscale
            
            if m == 2
                %             ET(idm{m},idm{m-1}) = -(nump(m)-1)*V{m}*nS(idm{m},idm{m-1})/nS(idm{m-1},idm{m-1});
                gradET(idm{m},idm{m-1}) = -EKinv{m}*ET(idm{m},idm{m-1})*Am{m};  %-nS(idm{m},idm{m-1})
            elseif m > 2
                gradET(idm{m},idm{m-1}) = -EKinv{m}*ET(idm{m},idm{m-1})*Am{m} ...  %-nS(idm{m},idm{m-1})
                    -Vinv{m}*ET(idm{m},idm{m-1})*((TVinvT{m-1}.*ELAMBDA{m-1})+diag(ELAMBDA{m-1}*diag(TVinvT{m-1}))/(nump(m)-1))/nump(m)/(nump(m)-3);
                %             -((nump(m)-1)*V{m}*nS(idm{m},idm{m-1})+ET(idm{m},idm{m-1})*(ELAMBDA{m-1}.*EK{m-1} ...
                %                 +((nump(m)-1)*TVinvT{m-1}.*ELAMBDA{m-1}+diag(ELAMBDA{m-1}*diag(TVinvT{m-1})))/nump(m)/(nump(m)-3)))/nS(idm{m-1},idm{m-1});
            end
        end
        
        gradET = sparse(idr_T,idc_T,gradET(idl_T) - nS(idl_T),p,p);
        sumgrad = sum(sum(gradET.^2));
        gradET = gradET + gradET';
        
        
        eta = etaT0;
        while 1
            ETtmp = ET + eta*gradET;
            [ELBOtmp,TVinvT,ETKinvT,ETKinvT2] = ELBOTree(Vinv,nump,ETtmp,EK,ELAMBDA,nS,idm,nscale);
            if ELBOtmp >= ELBO0 + 0.01*eta*sumgrad
                ET = ETtmp;
                break;
            end
            eta = eta/2;
        end
        eta_array2(kappa) = eta;
        
        diffT = ET(:) - ET0(:);
        if max(abs(diffT)) < tolm && sum(diffT.^2)/sum(ET0(:).^2) < tolr
            break;
        else
            ET0 = ET;
            ELBO0 = ELBOtmp;
        end
        
    end
    
    EKall = full(ET);
    for m = 1:nscale-1
        EKall(idm{m},idm{m}) = EK{m}+ETKinvT{m};
    end
    EKall(pc+1:end,pc+1:end) = EK{nscale};
    
    KCF = EKall(1:pc,1:pc);  %(1-eta)*KCF + eta*
    if max(max(abs(KCF-KCF'))) > 1e-4
        KCF = (KCF+KCF')/2;
    end
    hCF = -data*sparse(EKall(pc+1:end,1:pc));  %(1-eta)*hCF - eta*
    XC = hCF/KCF;
    nS(1:pc,1:pc) = XC'*XC+n*inv(KCF);
    nS(1:pc,pc+1:end) = XC'*data;
    nS(pc+1:end,1:pc) = nS(1:pc,pc+1:end)';
    
    
    
    b = 0;
    for m = 2:nscale %1:nscale
            
        EK2{m} = EK{m}.^2 + nu(m)*VARKdnu{m};

        if m == nscale
            EKsum2{m} = EK2{m};
        else
            EKsum2{m} = EK2{m}+2*EK{m}.*ETKinvT{m}+ETKinvT2{m};
        end
        d{m} = Eomega/2*EKsum2{m}(idl_lbd{m}); %(1-eta)*d{m} + eta*Eomega(m)/2*EKsum2{m}(idl_lbd{m});
        idd = find(d{m}>=10)';
        if ~isempty(idd)
            for i = idd
                expd{m}(i) = Lentz_Algorithm(d{m}(i));
            end
        end
        expd{m}(d{m}<10) = exp(d{m}(d{m}<10)).*expint(d{m}(d{m}<10));
        Elambda{m} = 1./(expd{m}.*d{m})-1;
%         btmp = btmp + sum(Elambda{m}.*EKsum2{m}(idl_lbd{m}))/2;
        b = b+sum(Elambda{m}.*EKsum2{m}(idl_lbd{m}))/2; %(1-eta)*b(m) + eta*sum(Elambda{m}.*EKsum2{m}(idl_lbd{m}))/2;
        if any(isinf(d{m}))   || any(d{m}<0)
            d{m}
        end
    end
    
    Eomega = a/b;
    for m = 2:nscale  %1:
        ELAMBDA{m}(idl_lbd{m}) = Eomega*Elambda{m}; %sparse(idr_lbd{m},idc_lbd{m},Eomega(m)*Elambda{m},nnodes(m+1),nnodes(m+1));
        ELAMBDA{m}(idu_lbd{m}) = ELAMBDA{m}(idl_lbd{m});
    end
    
    Am{2} = nS(idm{1},idm{1});
    for m = 3:nscale
        Am{m} = nS(idm{m-1},idm{m-1})+ELAMBDA{m-1}.*EK{m-1};
    end
    
%     b = (1-eta)*b + eta*btmp;
    
    if rem(kappa,10) == 0
        diffmax = 0; %abs(EK{1}-EK0{1});
        diffr = 0; %(EK{1}-EK0{1})^2;
        diff = 0; %EK0{1}^2;
        difgamma = 0;
%         diffmean = zeros(nscale,1);
        for m =  1:nscale
            diffm = EKall(idm{m},idm{m}) - EKall_old(idm{m},idm{m});
            diffmax = max(diffmax,max(abs(diffm(:))));
            diffr = diffr + sum((diffm(:)).^2);
            diff = diff + sum(sum(EKall_old(idm{m},idm{m}).^2));
            if m > 2
                gamma{m} = Elambda{m} ./ (1+Elambda{m});
                difgamma = max(difgamma, max(abs(gamma{m} - gamma_old{m})));
            end
        end
        diffr = sqrt(diffr/diff);
%         ELBOoverall = ELBO_overall_multi_oneomega(a1,b1,V,nu,ET,EK,EKsum2,EKinv,ELAMBDA,expd,nS,idm,nscale,n_nodes,n,a,b,d,KCF);
        fprintf('#no. of iterations =%d, difmax = %d, difrm = %d, difgamma = %d\n',kappa,diffmax,diffr,difgamma);
        if diffr < tolr && diffmax < tolm && difgamma < tolg %&& kappa > 500 && eta < eta0
            break;
        else
%             ELBOoverall0 = ELBOoverall;
            EKall_old = EKall;
            gamma_old = gamma;
        end
%         omega_thr = min(1e4,omega_thr*1.01);
%         c = max(0.5,c*0.99);
%         eta0 = max(1e-4,eta0*0.99);
%         tol_quic = max(1e-6,tol_quic*0.95);
    end
end

t = toc;
run_time = t;
fprintf("VWA-MonoGGM is done, elapsed time is %d seconds\n", t);

%%
fprintf("estimate adjacency matrix by thresholding lambda / (1 + lambda)...\n");
tic;
Kest = PostProcessing(ET, EK, EKinv, Elambda, Eomega, ELAMBDA, idm, nscale);
t = toc;
run_time = run_time + t;
fprintf("adjacency marix has been estimated, elapsed time is %d seconds\n", t);

%%
fprintf("start reestimating the non-zero elements...\n");
tic;
Kest = Kest - diag(diag(Kest));
Kest = Kest + diag(sum(abs(Kest)) + 0.1);
Kest = gaussEM(Kest, nSf / n);
t = toc;
fprintf("reestimating the non-zero elements is done, elapsed time is %d seconds\n", t);
run_time = run_time + t;

end

function E1b =  Lentz_Algorithm(x)
    epsilon1 = 1e-30;
    epsilon2 = 1e-7;
    f_prev = epsilon1;
    C_prev = epsilon1;
    D_prev = 0;
    delta = 2+epsilon2;
    j = 1;
    while (delta-1>=epsilon2 || 1-delta >= epsilon2)
        j = j+1;
        tmp1 = x+2*j-1;
        tmp2 = (j-1)^2;
        D_curr = 1/(tmp1-tmp2*D_prev);
        C_curr = tmp1-tmp2/C_prev;
        delta = C_curr*D_curr;
        f_curr = f_prev*delta;
        f_prev = f_curr;
        C_prev = C_curr;
        D_prev = D_curr;
    end
    E1b =  1/(x+1+f_curr);
end
