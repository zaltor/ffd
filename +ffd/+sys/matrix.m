function sys = matrix(K, block_size)
%ffd.sys.MATRIX    Create a FFD system using a K matrix
%   SYS = ffd.sys.MATRIX(K) creates an optical system specification for use
%   with FFD. K is a system matrix describing the linear propagation of
%   light from source spatial samples to spatial samples whose intensities
%   will eventually be collected to form measurements. K'*X will generate a
%   set of output field values for input field X. The system will be split
%   block-wise automatically into submatrices of at most 1048576 entries
%   (unless there are more than 1048576 rows)
%
%   SYS = ffd.sys.MATRIX(K, BLOCK_SIZE) is the same as above, except
%   instead of 1048576 entries, matrix K will be split block-wise into
%   BLOCK_SIZE entries each.

if ~exist('block_size','var')
    block_size = 1048576;
end

N = size(K,1);
M = size(K,2);
num_cols = max(1,floor(block_size/N)); % maximum number of columns
num_blocks = ceil(M/num_cols);
temp = 0:num_blocks;
temp = temp*(M/num_blocks);
temp = round(temp);
blocks = diff(temp);
assert(sum(blocks)==M);
sys = struct;
sys.parts = ffd.sys.parts_template(blocks);
sys.N = N;
sys.forward = @forward_helper;
sys.adjoint = @adjoint_helper;

    function Y = forward_helper(X,i)
        block = sys.parts(i);
        Y = K(:,block(1):block(2))'*X;
    end

    function X = adjoint_helper(Y,i)
        block = sys.parts(i);
        X = K(:,block(1):block(2))*Y;
    end


end

