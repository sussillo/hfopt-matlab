function [Anormed, Anorms] = normify(A)
% So sick of doing this over and over again. 
[N,M] = size(A);

Anorms = zeros(1,M);
Anormed = zeros(N,M);
for m = 1:M
    Anorms(m) = norm(A(:,m));
    Anormed(:,m) = A(:,m)/Anorms(m);
end

