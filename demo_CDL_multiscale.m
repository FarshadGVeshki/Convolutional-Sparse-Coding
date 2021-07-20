clear
clc
close all
rng('default')
%% Dictionary size
Kls = [6 6 6]; % numbers of filters of each size
Mls = [8 12 16]; % filter sizes

%% Load training data
P = 10; % number of training images
Pls = randi(40,[1 P]);
for p = 1:P
S0(:,:,p) = single(imread(['.\Data\IM_' num2str(Pls(p)) '.png']))/255;
end

%% Filter input images and compute highpass images
h = 16;
Sh = S0 - ifft2(fft2(S0).*fft2(ones(h)/h^2,size(S0,1),size(S0,2)),'symmetric');

%% Construct initial dictionary
D0 = initdict(Mls,Kls,0);

%% CDL
lamb = 0.05;
opt.MaxIter = 300;
[~,D,Res] = CDL_multiscale(D0,Sh,lamb,Kls,Mls,opt);

% save('dict_multiscale','D')

%% plotting
figure(124)
semilogy(Res.iterinf(:,end),Res.iterinf(:,2),'linewidth',2)
grid on
xlabel('time')
xlabel('fval')

figure(112)
imshow(dict2image(D),[]) % learned filters
