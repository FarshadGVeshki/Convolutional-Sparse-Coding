function I = dict2image(D,dcfilter)
if nargin <2
    dcfilter = 0;
end

[m1,m2,K] = size(D);
D = D-min(D,[],1:2);
D = D./max(D,[],1:2);

if dcfilter == 1
    ddc = D(:,:,1);
    ddc(find(ddc)) = 0.5;
    D(:,:,1) = ddc;
end

Ncol = ceil(sqrt(K));

I = [];

r = 1;
c = 1;
for k = 1:K
    I(r:r+m1-1,c:c+m2-1) = D(:,:,k);
    if rem(k,Ncol) ==0
        c = 1;
        r = r + m1 + 1;
    else
        c = c+m2+1;
    end
end

I = padarray(I,[1 1],'both');
end
