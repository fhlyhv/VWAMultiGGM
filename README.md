# VWAMultiGGM
Variational Wishart Approximation for Multiscale/Multiresolution Graphical Models

Varational Wishart Approximation for Monoscale Graphical Model Selection

The MATLAB code package implements the variational Wishart approximation algorithm to learn multiscale graphical models as described in the following paper:

 H. Yu, L. Xin, J. Dauwels, “Variational Wishart Approximation for Graphical Model Selection: Monoscale and Multiscale Models”, IEEE Transactions on Signal Processing, vol. 67, no. 24, pp. 6468 – 6482, 2019.

The main script is test.m. To test your own datasets, please replace XDat in polycovdata.mat by your own datasets and provide the adjacency matrix of the latent tree and the number of nodes in each scale. Note that XDat is a nxp matrix, where n is the sample size and p is the dimension.
