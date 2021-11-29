clear
clc
close all

%% load dictionary
load('dictK16m8.mat')
% load('dict_multiscale16.mat')
%% highpassed input image
hsize = size(D,1);

S = imread('.\Data\lena_std.tif');
S = double(rgb2gray(S))/255;
Smean = conv2(S,ones(hsize)/hsize^2,'same');
Sh = S - Smean;
[H, W, ~] = size(Sh);
%%
lamb = 0.05;
[X,Res] = CSC_unconstrained(D,Sh,lamb);

%% 


figure(1)
semilogy(Res.iterinf(:,end),Res.iterinf(:,2))
xlabel('time')
ylabel('fval')
grid on

%% reconstruction

S_rec = ifft2(sum(fft2(D,size(X,1),size(X,2)).*fft2(X),3),'symmetric') + Smean;

figure(2)
subplot(121)
imshow(S,[])
title('Original')
subplot(122)
imshow(S_rec, [])
title('Reconstructed')
