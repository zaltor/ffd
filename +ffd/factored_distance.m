function n = factored_distance(X1, X2, partition)
%FACTORED_DISTANCE    Compute norm(X1*X1'-X2*X2','fro') memory-efficiently
%   N = FACTORED_DISTANCE(X1, X2) will split X1 and X2 into blocks and
%   compute norm(X1*X1'-X2*X2','fro') in a memory-efficient fashion. It has
%   a special case for size(X1,2)==size(X2,2)==1, which is very fast.

N = size(X1,1);
if size(X2,1) ~= N
    error('factored_distance:input', ...
          'X1 and X2 must have the same number of rows');
end

if ~exist('X2','var')
    X2 = sparse(N,1);
end

if ~exist('partition','var')
    partition = 104857600;
end

% shortcut if both X1 and X2 have only one column
if numel(X1) == N && numel(X2) == N
    aha = (X1'*X1);
    ahb = (X1'*X2);
    bhb = (X2'*X2);
    temp1 = (X2-(ahb/aha)*X1);
    temp2 = (X1-(conj(ahb)/bhb)*X2);
    n = sqrt((aha-bhb)^2+aha*(temp1'*temp1)+bhb*(temp2'*temp2));
    return;
end

block_size = ceil(partition/N);

blocks = ceil(N/block_size);

temp = zeros(blocks,1);
j=1;

for i=1:block_size:N
    block_start = i;
    block_end = min(block_start+block_size-1,N);
    temp2 = X1*X1(block_start:block_end,:)' - X2*X2(block_start:block_end,:)';
    temp(j) = norm(temp2(:),2);
    j = j + 1;
end

n = norm(temp,2);

end

