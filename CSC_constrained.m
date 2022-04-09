function [X, res] = CSC_constrained(D, S, Eps, opts)
% Efficient algorithm for constrained convolutional sparse coding (CSC)
%
% Inputs:
% D:    initial dictionary of K filters of size m1 by m2
% S:    image of size H by W
% Eps: upperbound on approximation error (energy)
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
%
% reference: 
%
% Farsad G. Veshki and Sergiy Vorobyov. Efficient ADMM-based Algorithms for
% Convolutional Sparse Coding, IEEE Signal Processing Letters, 2021
%% parameters
[H,W] = size(S);
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
if ~isfield(opts,'eAbs')
    opts.eAbs = 1e-4;
end
if ~isfield(opts,'eRel')
    opts.eRel = 1e-4;
end
if ~isfield(opts,'Xinit')
    opts.Xinit = zeros(H,W,K);
end
if ~isfield(opts,'Uinit')
    opts.Uinit = zeros(H,W,K);
end
if ~isfield(opts,'lamb')
    opts.lamb = 1;
end
if ~isfield(opts,'relaxParam')
    opts.relaxParam = 1.8;
end
if ~isfield(opts,'AutoRho')
    opts.AutoRho = 1;
end

%% initialization
alpha = opts.relaxParam;
Sf = fft2(S);
X = opts.Xinit; % sparse code
U = opts.Uinit; % scaled dual variable
rho = opts.rho;
lamb = opts.lamb;
Nx = numel(X);
eAbs = opts.eAbs;
eRel = opts.eRel;
epri = 0;
edua = 0;

MaxIter = opts.MaxIter;
r = inf; s = inf;
res.iterinf = [];
itr = 1;

mu = 10; % varying rho parameter
tau = 1.2;

tsrt = tic;

Df = fft2(D,H,W);
SDD = sum(abs(Df).^2,3);
nu = 1;

%% ADMM iterations
while itr <= MaxIter && (r > epri || s > edua)
    
    %_________________________Z update______________________________
    [Z, nu, nitr_bisec]  = Z_update(fft2(X-U),Df,Sf,SDD,nu,Eps,H*W) ;
    Zr = alpha *Z + (1-alpha)*X;
    
    %_________________________X update______________________________
    Xprv = X;
    X = sfthrsh(Zr+U, lamb/rho);
    
    %_________________________U update______________________________
    U = Zr - X + U;
    
    %_______________________________________________________________
    %_________________________residuals_____________________________
    nZ = norm(Z(:)); nX = norm(X(:)); nU = norm(U(:));
    r = norm(vec(X-Z)); % primal residulal
    s = rho*norm(vec(Xprv-X)); % dual residual
    epri = sqrt(Nx)*eAbs+max(nX,nZ)*eRel;
    edua = sqrt(Nx)*eAbs+rho*nU*eRel;
    
    titer = toc(tsrt);
    
    %_________________________progress_______________________________
    JL1 =  sum(abs(X(:))); % L_1 norm
    rPow = sum(vec(abs(sum(Df.*fft2(X),3)-Sf).^2))/(H*W);% residual power
    res.iterinf = [res.iterinf; [itr rPow/2+lamb*JL1 rPow JL1 r s titer]];
    res.U = U;
    res.nu(itr,1) = nu;
    res.BS_iters(itr,1) = nitr_bisec;
    %_________________________rho update_____________________________
    
    if opts.AutoRho && itr ~= 1
        [rho,U] = rho_update(rho,r,s,mu,tau,U);
    end
    
    res.rho(itr,1) = rho;
    %_______________________________________________________
    itr = itr + 1;
end

end

function y = sfthrsh(x, kappa)
y = sign(x).*max(0, abs(x) - kappa);
end

function [rho,U] = rho_update(rho,r,s,mu,tau,U)
rhomlt = sqrt(r/(s*mu));
if rhomlt < 1, rhomlt = 1/rhomlt; end
if rhomlt > 100, rhomlt = 100; end 

rsf = 1;
if r > mu*tau*s, rsf = rhomlt; end
if s > (tau/mu)*r, rsf = 1/rhomlt; end
rhot = rsf*rho;
if rhot>1e-4
    rho = rhot;
    U = U/rsf;
end

end

function [Z, nu, nitr_bs]  = Z_update(Wf,Df,Sf,SDD,nu,Eps,N)
Rf = Sf - sum(Wf.*Df,3); % residual update
nR = norm(Rf(:))/sqrt(N);

if nR<sqrt(Eps)
    Z = ifft2(Wf,'symmetric');
    nitr_bs = 0;
else
    [nu, nitr_bs] = Bisection(nu,Rf,SDD,Eps*N,N,100);
    Zf = Wf + (conj(Df)./(SDD+nu)).*Rf; % Z update
    Z  = ifft2(Zf,'symmetric');
end
end


function [nu, i] = Bisection(nu_old,Rf,PSD,Eps,N,niters)
a = nu_old*0.9;
b = nu_old*1.1;
while norm(vec((Rf)./(PSD+a)))^2 > Eps/a^2
    b = a;
    a = a/2;
end
while norm(vec((Rf)./(PSD+b)))^2 < Eps/b^2
    a = b;
    b = 2*b;
end

for i=1:niters
    c = (a+b)/2;
    e = c^2*norm(vec(Rf./(PSD+c)))^2;
    if abs(e-Eps)/N>1e-4
        if e < Eps
            a = c;
        else
            b = c;
        end
    else
        break
    end
end
nu = c;
end

function x = vec(y)
x = y(:);
end
