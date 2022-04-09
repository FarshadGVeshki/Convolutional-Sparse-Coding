function [X, res] = CSC_unconstrained(D, S, lamb, opts)
% Efficient algorithm for unconstrained convolutional sparse coding (CSC)
%
% Inputs:
% D:    initial dictionary of K filters of size m1 by m2
% S:    P stacked image of size H by W (S is H by W by P)
% lamb: l1-norm regularization parameter
% opt:  optional parameters
%
% Outputs:
% X:    sparse coefficient maps
% res: results (iterations details)
%
% Optional parameters:
%
% opts.MaxIter:  maximum number of iterations (default 500)
% opts.rho: ADMM penalty parameter for csc step (default 10) 
% opts.AutoRho: varying penalty parameter (ADMM extention) (default 1 (enabled))
% opts.relaxParam: relaxation parameter (ADMM extention) (default 1.8)
% opts.Xinit:   initial sparse codes (default zeros)
% opts.Uinit:   initial lagrange variables (default zeros)
% opts.eAbs: absolute tolerance for stopping criteria (deafult 10^-3)
% opts.eRel: relative tolerance for stopping criteria (deafult 10^-3)
%
% reference: 
%
% Farsad G. Veshki and Sergiy Vorobyov. Efficient ADMM-based Algorithms for
% Convolutional Sparse Coding, IEEE Signal Processing Letters, 2021
%% parameters

[H,W,P] = size(S);
S = reshape(S,[H W 1 P]);
K = size(D,3);

if nargin < 4
    opts = [];
end
if ~isfield(opts,'MaxIter')
    opts.MaxIter = 300;
end
if ~isfield(opts,'rho')
    opts.rho = 10;
end
if ~isfield(opts,'AutoRho')
    opts.AutoRho = 1;
end
if ~isfield(opts,'eAbs')
    opts.eAbs = 1e-4;
end
if ~isfield(opts,'eRel')
    opts.eRel = 1e-4;
end
if ~isfield(opts,'Xinit')
    opts.Xinit = zeros(H,W,K,P);
end
if ~isfield(opts,'Uinit')
    opts.Uinit = zeros(H,W,K,P);
end
if ~isfield(opts,'relaxParam')
    opts.relaxParam = 1.8;
end

%% initialization
alpha = opts.relaxParam;
Sf = fft2(S);
X = opts.Xinit; % sparse code
U = opts.Uinit; % scaled dual variable
rho = opts.rho;
eAbs = opts.eAbs;
eRel = opts.eRel;
epri = 0;
edua = 0;


rho_flg = 0;
MaxIter = opts.MaxIter;
r = inf; s = inf;
res.iterinf = [];

mu = 5; % varying rho parameter
tau = 1.2; % varying rho parameter

itr = 1;
vec = @(x) x(:);

tsrt = tic;

Df = fft2(D,H,W);
SDD = sum(abs(Df).^2,3);
Cf = conj(Df)./(SDD+rho);
Nx = numel(X);

%% ADMM iterations
while itr <= MaxIter && (r > epri || s > edua)
    
    %_________________________Z update______________________________
    [Zf,Cf,rho_flg]  = Z_update(fft2(X-U),Cf,Df,Sf,SDD,rho,rho_flg) ;
    Z  = ifft2(Zf,'symmetric');
    Zr = alpha * Z + (1-alpha)*X;
    
    %_________________________X update______________________________
    Xprv = X;
    X = sfthrsh(Zr+U, lamb/rho);
    
    %_________________________U update______________________________
    U = Zr - X + U;
    
    %_______________________________________________________________
    %_________________________residuals_____________________________
    nX = norm(X(:)); nZ = norm(Z(:)); nU = norm(U(:));
    r = norm(vec(X-Z)); % primal residulal
    s = rho*norm(vec(Xprv-X)); % dual residual
    epri = sqrt(Nx)*eAbs+max(nX,nZ)*eRel;
    edua = sqrt(Nx)*eAbs+rho*nU*eRel;
    
    titer = toc(tsrt);
    
    %_________________________progress_______________________________
    JL1 =  sum(abs(X(:))); % L_1 norm
    rPow = sum(vec(abs(sum(Df.*fft2(X),3)-Sf).^2))/(H*W); % residual power
    fval = rPow/2 + lamb*JL1; % functional value
    res.iterinf = [res.iterinf; [itr fval rPow JL1 r s rho titer]];
    res.U = U;
    %_________________________rho update_____________________________
    if opts.AutoRho && itr ~= 1
        [rho,U,rho_flg] = rho_update(rho,r,s,mu,tau,U);
    end
    res.rho(itr,1) = rho;
    %_______________________________________________________________
    itr = itr + 1;
end
end

function y = sfthrsh(x, kappa)
y = sign(x).*max(0, abs(x) - kappa);
end

function [Zf,Cf,rho_flg]  = Z_update(Wf,Cf,Df,Sf,SDD,rho,rho_flg)
Rf = Sf - sum(Wf.*Df,3); % residual update
if rho_flg == 0
    Zf = Wf + Cf.*Rf; % Z update
else
    rho_flg = 0;
    Cf = conj(Df)./(SDD+rho);
    Zf = Wf + Cf.*Rf; % Z update
end
end

function [rho,U,rho_flg] = rho_update(rho,r,s,mu,tau,U)
rhomlt = sqrt(r/(s*mu));
if rhomlt < 1, rhomlt = 1/rhomlt; end
if rhomlt > 100, rhomlt = 100; end

rsf = 1;
if r > mu*tau*s, rsf = rhomlt; end
if s > (tau/mu)*r, rsf = 1/rhomlt; end
rhot = rsf*rho;
if rsf ~= 1
    rho_flg =  1;
else
    rho_flg =  0;
end
if rhot>1e-4
    rho = rhot;
    U = U/rsf;
end

end
