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
