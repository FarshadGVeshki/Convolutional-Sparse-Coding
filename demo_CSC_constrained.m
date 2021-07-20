clear
clc
close all

%% load dictionary
load('dictK16m8.mat')

%% highpassed input image
hsize = size(D,1);

S = imread('.\Data\lena_std.tif');
S = double(rgb2gray(S))/255;
Smean = conv2(S,ones(hsize)/hsize^2,'same');
Sh = S - Smean;
[H, W, ~] = size(Sh);

%% parameters and setting
opts.MaxIter = 200;
opts.AutoRho= 0;

%% Unconstrained CSC
lamb = 0.05;

[X,Res_uncons] = CSC_unconstrained(D,Sh,lamb,opts);

S_rec_uncons = ifft2(sum(fft2(D,H,W).*fft2(X),3),'symmetric');
rt_uncons = Res_uncons.iterinf(end,end); % runtime unconstrained CSC
Err_uncons = norm(Sh(:)-S_rec_uncons(:))^2; % approximation error unconstrained CSC
L1_uncons = sum(abs(X(:))); % L1-norm unconstrained CSC

%% Constrained CSC
Eps = Err_uncons;
opts.lamb = lamb;

[X_cons,Res_cons] = CSC_constrained(D, Sh, Eps,opts);

S_rec_cons = ifft2(sum(fft2(D,H,W).*fft2(X_cons),3),'symmetric');
rt_cons = Res_cons.iterinf(end,end);
Err_cons = norm(Sh(:)-S_rec_cons(:))^2; % approximation error constrained CSC
L1_cons = sum(abs(X_cons(:))); % L1-norm constrained CSC

%% printing results
fprintf('Results: \n')
fprintf('%s %11s %10s %12s \n', 'CSC method','L1_norm','Error','runtime')
fprintf('%s %10s %10s %10s \n', 'Unconstrained',num2str(L1_uncons),num2str(Err_uncons),num2str(rt_uncons))
fprintf('%s %12s %10s %10s \n', 'Constrained',num2str(L1_cons),num2str(Err_cons),num2str(rt_cons))


