function D = initdict(M,K,dcfilter)
% generating initial multiscale dictionary
% M is array of filter sizes
% K is array of number filters of each size
% number of elements in M an K must be equal
% if dcfilter is 1, first filter is returned is dc filter


s = max(M);
D =[];

for i = 1:length(M)
d = randn(M(i),M(i),K(i));
d  = (d)./sqrt(sum(d.^2,1:2));
d = padarray(d,[s-M(i)  s-M(i)],'post');
D = cat(3,D,d);
end

if dcfilter == 1
    D(:,:,1) = padarray(1/M(1)*ones(M(1)),[s-M(1)  s-M(1)],'post');
end


D = single(D);
end