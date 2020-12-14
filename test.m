load('policovdata.mat')

[V,nu,EKall,Kest] = VWA_Multi_GGM(XDat, nnodes, T);

figure;
subplot(1, 2, 1); 
imagesc(Ptrue);
title('true cov');
subplot(1, 2, 2);
Sest = inv(Kest);
imagesc(Sest(end - nnodes(end) + 1 : end, end - nnodes(end) + 1 : end));
title('cov estiamted by VWA-MultiGGM');

