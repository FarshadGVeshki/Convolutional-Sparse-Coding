function [X,D, res] = CDL_multiscale(D, S, lamb, Kls, Mls, opts)
% Efficient algorithm for onvolutional dictionary learning (CDL) (with multiscale filters)
%
% Inputs:
% D:    initial dictionary of K filters of size m1 by m2
% S:    P stacked image of size H by W (S is H by W by P)
% Kls:  list of number of filters of each size
% Mls:  list of filter sizes  (Mls and Kls must be equal)
% lamb: l1-norm regularization parameter
% opt:  optional parameters
%
% Outputs:
% X:    sparse coefficient maps
% D:    the learned dictionary
% res: results (iterations details)
%
% Optional parameters:
%
% opts.MaxItr:  maximum number of iterations (default 500)
% opts.csc_iters: number of internal csc step iterations (default 1) 
% opts.cdl_iters: number of internal dict update step iterations (default 1) 
% opts.rho: ADMM penalty parameter for csc step (default 10) 
% opts.sig: ADMM penalty parameter for cdl step (default 10) 
% opts.AutoRho: varying penalty parameter (ADMM extention) (default 0 (disabled))
% opts.AutoSig: varying penalty parameter (ADMM extention) (default 0 (disabled))
% opts.RhoUpdateCycle: rho update cycle (default 10)
% opts.SigUpdateCycle: sig update cycle (default 10)
% opts.relaxParam: relaxation parameter (ADMM extention) (default 1.8)
% opts.Xinit:   initial sparse codes (default zeros)
% opts.Uinit:   initial lagrange variables (default zeros)
% opts.Vinit:   initial lagrange variables (default zeros)
% opts.dcfilter:    the first filter is DC filter and is kept unchanges (default 0)
%
% reference: 
%
% Farsad G. Veshki and Sergiy Vorobyov. Efficient ADMM-based Algorithms for
% Convolutional Sparse Coding. 2021

%%
if length(Mls) ~= length(Kls)
    error('Numbers of elements in K and M are diffrent.')
end

%% parameters

[H,W,P] = size(S);
[m,~,K] = size(D);
S = reshape(S,[H W 1 P]);




if nargin < 4
    opts = [];
end
if ~isfield(opts,'MaxIter')
    opts.MaxIter = 1000;
end
if ~isfield(opts,'csc_iters')
    opts.csc_iters = 1;
end
if ~isfield(opts,'cdl_iters')
    opts.cdl_iters = 1;
end
if ~isfield(opts,'rho')
    opts.rho = 10;
end
if ~isfield(opts,'sig')
    opts.sig = 10;
end
if ~isfield(opts,'AutoRho')
    opts.AutoRho = 1;
end
if ~isfield(opts,'AutoSig')
    opts.AutoSig = 1;
end
if ~isfield(opts,'AutoSig')
    opts.AutoSig = 1;
end
if ~isfield(opts,'RhoUpdateCycle')
    opts.RhoUpdateCycle = 1;
end
if ~isfield(opts,'SigUpdateCycle')
    opts.SigUpdateCycle = 1;
end
if ~isfield(opts,'Xinit')
    opts.Xinit = zeros(H,W,K,P,'single');
end
if ~isfield(opts,'Uinit')
    opts.Uinit = zeros(H,W,K,P,'single');
end
if ~isfield(opts,'Vinit')
    opts.Vinit = zeros(H,W,K,P,'single');
end
if ~isfield(opts,'relaxParam')
    opts.relaxParam = 1.8;
end
if ~isfield(opts,'eAbs')
    opts.eAbs = 1e-3;
end
if ~isfield(opts,'eRel')
    opts.eRel = 1e-3;
end
if ~isfield(opts,'dcfilter')
    opts.dcfilter = 0;
end
%% initialization
alpha = opts.relaxParam;
Sf = fft2(S);
X = opts.Xinit; % sparse code
U = opts.Uinit; % scaled dual variable
V = opts.Vinit;
rho = opts.rho;
sig = opts.sig;
RhoUpdateCycle = opts.RhoUpdateCycle;
SigUpdateCycle = opts.SigUpdateCycle;


epri = opts.eAbs;
edua = opts.eAbs;

MaxIter = opts.MaxIter;
csc_iters = opts.csc_iters;
cdl_iters = opts.cdl_iters;

r_csc = inf; s_csc = inf; r_cdl= inf; s_cdl = inf;
res.iterinf = [];
mu = 5; % varying rho parameter
tau = 1.2; % varying rho parameter

vec = @(x) x(:);
itr = 1;

%% CDL CYCLES
tsrt = tic;

D = padarray(D,[H-m W-m],'post');
Df = fft2(D);
while itr<=MaxIter && (r_csc > epri || s_csc > edua || r_cdl > epri || s_cdl > edua)
    %%% ADMM iterations
    
    %% CSC
    for ttt = 1:csc_iters % default = 1
        Xprv = X;
        Z  = Z_update(fft2(X-U),Df,Sf,rho) ; % X update
        Zr = alpha * Z + (1-alpha)*X; % relaxation
        X = sfthrsh(Zr+U, lamb/rho); % Z update
        U = Zr - X + U; % U update
    end
    
    %% CDL
    for ttt = 1:cdl_iters % default = 1
        Dprv = D;
        G = G_update(fft2(X),Sf,sig,fft2(D-V));
        Gr = alpha * G + (1-alpha)*D; % relaxation
        
                
        frst = 1;
        for id = 1:length(Mls) 
        D(:,:,frst:frst+Kls(id)-1) = D_proj(  sum(Gr(:,:,frst:frst+Kls(id)-1,:)+V(:,:,frst:frst+Kls(id)-1,:),4)/P ,Mls(id),H,W); % projection on constraint set
        frst = frst+Kls(id);
        end
        if opts.dcfilter == 1
            D(:,:,1) = Dprv(:,:,1);
        end
        
        V = Gr - D + V;
    end
    %%
    Df = fft2(D);
    titer = toc(tsrt);
    %%
    
    %_________________________residuals CSC_____________________________
    nX = norm(Z(:)); nZ = norm(X(:)); nU = norm(U(:));
    r_csc = norm(vec(Z-X))/(max(nX,nZ)); % primal residulal
    s_csc = norm(vec(Xprv-X))/nU; % dual residual
    
    %_________________________residuals CDL_____________________________
    nG = norm(G(:)); nD = norm(D(:))*sqrt(P); nV = norm(V(:));
    r_cdl = norm(vec(G-D))/(max(nG,nD)); % primal residulal
    s_cdl = (norm(vec(Dprv-D))/nV)*sqrt(P); % dual residual
    
    %_________________________rho update_____________________________
    if opts.AutoRho && rem(itr,RhoUpdateCycle)==0
        [rho,U] = rho_update(rho,r_csc,s_csc,mu,tau,U);
    end
    
    %_________________________sig update_____________________________
    if opts.AutoSig && rem(itr,SigUpdateCycle)==0
        [sig,V] = rho_update(sig,r_cdl,s_cdl,mu,tau,V);
    end
    
    %_________________________progress_______________________________
    rPow = sum(vec(abs(sum(Df.*fft2(X),3)-Sf).^2))/(2*H*W); % residual power
    L1 = sum(abs(X(:))); % l1-norm
    fval = rPow + lamb*L1; % functional value
    res.iterinf = [res.iterinf; [itr fval rPow L1 r_csc s_csc r_cdl s_cdl rho sig titer]];
    
    
    itr = itr+1;
end
D = D(1:m,1:m,:);
end

function y = sfthrsh(x, kappa) % shrinkage
y = sign(x).*max(0, abs(x) - kappa);
end

function Z  = Z_update(Wf,Df,Sf,rho)
C = conj(Df)./(sum(abs(Df).^2,3)+rho);
Rf = Sf - sum(Wf.*Df,3); % residual update
Zf = Wf + C.*Rf; % X update
Z  = ifft2(Zf,'symmetric');
end

function G = G_update(Xf,Sf,sig,Wf)
C = conj(Xf)./(sum(abs(Xf).^2,3)+sig);
Rf = Sf - sum(Xf.*Wf,3); % residual update
Gf = Wf + C.*Rf;
G = ifft2(Gf,'symmetric');
end

function D = D_proj(D,m,H,W) % projection on unit ball
D = padarray(D(1:m,1:m,:,:),[H-m W-m],'post');
D  = D./max(sqrt(sum(D.^2,1:2)),0);
end


function [rho,U] = rho_update(rho,r,s,mu,tau,U)
% varying penalty parameter
a = 1;
if r > mu*s
    a = tau;
end
if s > mu*r
    a = 1/tau;
end
rho_ = a*rho;
if rho_>1e-4
    rho = rho_;
    U = U/a;
end
end