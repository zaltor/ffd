function y = forward(KH, X, C)
%ffd.FORWARD    Compute the forward transform and return "measurements"
%   Y = ffd.FORWARD(KH, X) will propagate the set of generalized coherence
%   modes given by X through an optical system specified by KH and return
%   a set of measurement intensities in Y.
%
%   Y = ffd.FORWARD(KH, X, C) will coagulate intensities into measurements
%   using C

% determine M and N
N = size(KH,2);
M = size(KH,1);

if N ~= size(X,1)
    error('forward:sysmismatch','X needs to have %d rows', N);
end

if ~exist('C','var')
    C = linops.Identity(M);
end

% create the output
y = zeros(M,1);

% fill in the output, part by part
for yIdx=1:KH.rowBlocks
    KHX_block = 0;
    for xIdx=1:KH.colBlocks
        KHX_block = KHX_block + KH.forward(yIdx,xIdx,X(KH.colFirst(xIdx):KH.colLast(xIdx),:));
    end
    y(KH.rowFirst(yIdx):KH.rowLast(yIdx)) = C.forward(yIdx,yIdx,sum(KHX_block.*conj(KHX_block),2));
end