clear
clc
close all
rng('default')
%% Dictionary size
K = 16; % number of filters
m = 8; % filter size

%% Load training data
P = 4; % number of training images
Pls = randi(40,[1 P]);
for p = 1:P
    S0(:,:,p) = single(imread(['.\Data\IM_' num2str(Pls(p)) '.png']))/255;
end

%% Construct initial dictionary
dcfilter = 0;
D0 = initdict(m,K,dcfilter);

%% Filter input images and remove local mean (highpass filter)
Sh = S0 - ifft2(fft2(S0).*fft2(ones(m)/m^2,size(S0,1),size(S0,2)),'symmetric'); % use this when dcfilter is 0
% Sh = S0; % use this when dcfilter is 1

%% CDL
lamb = 0.05;
opt.MaxIter = 100;
opt.dcfilter = dcfilter;
[~,D,Res] = CDL(D0,Sh,lamb,opt);
% [~,D,Res] = CDL_mtx_inv(D0,Sh,lamb,opt);

% save('dict','D')

%% plotting
figure(124)
semilogy(Res.iterinf(:,end),Res.iterinf(:,2),'linewidth',2)
grid on
xlabel('time')
xlabel('fval')

figure(112)
imshow(dict2image(D,dcfilter),[]) % learned filters
