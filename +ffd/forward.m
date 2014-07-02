function y = forward(sys, X)
%ffd.FORWARD    Compute the forward transform and return "measurements"
%   Y = ffd.FORWARD(SYS, X) will propagate the set of generalized coherence
%   modes given by X through an optical system specified by SYS and return
%   a set of measurement intensities in Y.

% determine M and N
N = sys.N;
M = 0;
for i=1:sys.parts()
    block = sys.parts(i);
    M = max(M,max(block));
end

if N ~= size(X,1)
    error('forward:sysmismatch','sys and X do not share the same N');
end

% create dummy gather operations if necessary
try
    sys.gather(sys.scatter(zeros(M,0),1),1);
    sysgather = @(y,i) sys.gather(y,i);
catch err
    sysgather = @(y,i) y;
end

% create the output
y = zeros(M,1);

% fill in the output, part by part
for i=1:sys.parts()
    block = sys.parts(i);
    temp = sys.forward(X,i);
    y(block(1):block(2)) = sysgather(sum(temp.*conj(temp),2),i);
end