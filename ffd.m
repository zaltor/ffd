function [X, iterations] = ffd(y, KH, varargin)
%FFD    Coherence retrieval using factored form descent
%   X = FFD(Y, KH) computes a set of (generalized) coherence modes from
%   intensity measurements and a known linear optical system. The optical
%   system is specified by a linear operator KH, an object of any subclass
%   of linops.Blockwise, which maps the field at each source sample
%   position to the field at each intensity measurement position, i.e.
%   giving the optical system's coherent impulse response. FFD will attempt
%   to minimize the least squared error in intensity measurements. The
%   result X is an N-by-N complex matrix where N is the number of spatial
%   samples in the source. Each column of X is a (generalized) coherence
%   mode, and Y is an M-by-1 vector containing the intensity measurements.
%
%   X = FFD(Y, KH, 'sigma', SIGMA) will instead minimize the weighted least
%   squared error in intensity measurements.  SIGMA is a vector with the
%   same size as Y.  The squared error corresponding to measurement Y(m) is
%   scaled by SIGMA(m)^-2.
%
%   [X, ITERATIONS] = FFD(...) stores per-iteration numbers such as the
%   merit function value, the RMS error in the intensity, etc. in the
%   structure ITERATIONS.
%
%   Merit Function
%   ==============
%
%   FFD works by reducing the value of a merit function one iteration at a
%   time. If the 'Q' regularization option is not specified, the merit
%   function can be computed in the following way (for more information
%   about the other parameters, please see the Optimization Options section
%   below):
%
%      fval = 0;
%      yBlocks = KH.rowBlocks;
%      xBlocks = KH.colBlocks;
%      yIdx1 = KH.rowFirst(1:yBlocks);
%      yIdx2 = KH.rowLast(1:yBlocks);
%      xIdx1 = KH.colFirst(1:xBlocks);
%      xIdx2 = KH.colLast(1:xBlocks);
%      for i=1:yBlocks
%          yprime = 0;
%          for j=1:xBlocks
%              xhat = KH.forward(i,j,X(xIdx1(j):xIdx2(j),:));
%              yprime = yprime + sum(xhat.*conj(xhat),2);
%          end
%          Cyprime = C.forward(i,i,yprime);
%          fval = fval + sum(W(yIdx1(i):yIdx2(i)).^2 .* ...
%                            (Y(yIdx1(i):yIdx2(i))-Cyprime).^2);
%      end
%
%   Optimization Options
%   ====================
%
%   FFD allows for many options to be specified at the command-line in the
%   form of pairs of parameter name and value:
%
%   FFD(Y, KH, 'param1', VALUE1, 'param2', VALUE2, ...)
%
%   Alternatively, a struct with field names equal to the parameter names
%   and field values equal to the parameter values can be passed in as well
%   using the parameter name 'opts': 
%
%   FFD(Y, KH 'opts', OPTS)
%
%   (when 'opts' is specified, any other options are ignored)
%
%   The following is a complete list of acceptable options:
%
%      FFD(..., 'imax', IMAX, ...) will cause FFD to run for at most
%      IMAX iterations. The default value is 1000.
%
%      FFD(..., 'B', B, ...) sets an assumed upper bound for
%      norm(X-X_THE,'fro').  The default value is Inf.
%
%      FFD(..., 'tau', TAU, ...) sets a termination threshold. FFD can
%      choose to terminate when it can prove that the merit function value
%      is within TAU_PRIME of the global minimum, assuming B is correct.
%      TAU_PRIME is TAU multiplied by the merit function value for X=0.
%      The default value is 0, i.e. no premature termination.
%
%      FFD(..., 'sigma', SIGMA, ...) specifies noise standard deviation for
%      each measurement and hence must be the same size as Y.  The default
%      value is a vector of ones.
%
%      FFD(..., 'R', R, ...) will restrict the solution to R modes.
%      Convergence guarantees are still an open topic of research for this
%      situation. The default value is N.
%
%      FFD(..., 'X1', X1, ...) will use the specific value X1 as the
%      initial value.  Otherwise, FFD uses zero as the initial value and
%      computes an initial descent direction based on factoring D.
%
%      FFD(..., 'Jthe', JTHE, ...) will effectively compute 
%      norm(X*X'-JTHE, 'fro')/N at each iteration, storing into JERRS. 
%
%      FFD(..., 'Xthe', XTHE, ...) will effectively compute
%      norm(X*X'-XTHE*XTHE', 'fro')/N at each iteration, storing into
%      JERRS. If the JERRS output variable is assigned (i.e. used), then at
%      least either 'Jthe' or 'Xthe' must be specified.
%
%      FFD(..., 'verbose', false, ...) will prevent FFD from printing out
%      progress messages.
%
%      FFD(..., 'verbose', true, ...) will enable progress messages.
%
%      FFD(..., 'callback', CALLBACK, ...) will invoke CALLBACK(ENV)
%      before the first iteration and after each iteration, where ENV holds
%      all the current state variables, etc.
%
%      FFD(..., 'descent', ffd.descent.Steepest, ...) will use steepest
%      descent as the step direction before conjugate gradients
%
%      FFD(..., 'cg', false, ...) will disable conjugate gradient and cause
%      the step direction to be simply the (preconditioned) gradient.
%
%      FFD(..., 'Q', Q, ...) will add an energy minimization term
%      trace((Q*X)*(Q*X)') to the merit function, where Q is a
%      linops.Blockwise object.
%
%      FFD(..., 'C', C) will cause FFD to minimize the (weighted) least
%      squared error in linear combinations of point-wise intensity
%      measurements, where C is a linops.Blockwise object. That is, if Z is
%      a vector of the intensity values derived from the current iterate,
%      then the merit function is equal to norm(Y-C*Z,2)^2. Note: C must be
%      block-wise diagonal and the row partitioning pattern for KH must be
%      the same as the row and column partitioning patterns for C.
%
%      FFD(..., 'rs', RS, ...) will force FFD to use a specific RandStream
%      for generating random numbers (i.e. for setting the initial value).
%      If unspecified, FFD will use the global random number generator.
%
%      FFD(..., 'maskJ', MASKJ, ...) will force FFD to only compute
%      JERRS based on certain elements of J. MASKJ is a logical N-by-1
%      array and only rows and columns in J corresponding to values of true
%      in MASKJ are used in JERRS calculation.
%
%      FFD(..., 'masky', MASKY, ...) will force FFD to only compute
%      YERRS based on certain elements of Y. MASKY is a logical M-by-1
%      array (i.e. the same size as Y), and an element is true if the
%      corresponding element in Y is to be used for RMS error calculation.

% set up a shared data store
env = ffd.Environment;

% set up constants (with respect to iteration)
env.consts = struct;

% check for valid y and extract size
if size(y,1) ~= numel(y)
    error('ffd:input:y','y must be an M-by-1 vector');
end
env.consts.M = size(y,1);
env.consts.y = y;
clear y;

% check for valid KH and extract size and partitioning
if ~isa(KH, 'linops.Blockwise')
    error('ffd:input:KH','KH must be a linops.Blockwise object');
end
env.consts.N = size(KH,2);
env.consts.xBlocks = KH.colBlocks;
env.consts.xIdx1 = KH.colFirst(1:env.consts.xBlocks);
env.consts.xIdx2 = KH.colLast(1:env.consts.xBlocks);
env.consts.KH = KH;
clear KH;

% get options
env.opts = struct;
env.opts.imax = 1000;
env.opts.sigma = [];
env.opts.B = Inf;
env.opts.tau = 0;
env.opts.X1 = [];
env.opts.R = env.consts.N;
env.opts.Jthe = [];
env.opts.Xthe = [];
env.opts.verbose = true;
env.opts.descent = ffd.descent.Equalized;
env.opts.callback = ffd.callback.Status(1);
env.opts.cg = true;
env.opts.rs = RandStream.getGlobalStream;
env.opts.C = linops.Identity(size(env.consts.KH,1),env.consts.KH.rowSplits);
env.opts.Q = [];
env.opts.masky = true(env.consts.M,1);
env.opts.maskJ = true(env.consts.N,1);

% set to true to have initial X0 be a "hint" for starting direction instead 
% of starting value
env.opts.X1seed = false; 

% grab the options
env.opts = mhelpers.getopts(env.opts, varargin{:});

% size of X
env.consts.Xsize = [env.consts.N, env.opts.R];

% precalculate sigma^-2
if isempty(env.opts.sigma)
    env.consts.w2 = true(env.consts.M,1);
else
    if numel(env.opts.sigma) ~= env.consts.M || size(env.opts.sigma,1) ~= env.consts.M
        error('ffd:input:sigma:size', 'sigma must be an M-by-1 vector');
    end
    if any(imag(env.opts.sigma(:))) || any(env.opts.sigma(:)<=0)
        error('ffd:input:sigma:type', 'sigma must be all positive');
    end
    env.consts.w2 = env.opts.sigma.^-2;
end

% compute actual termination threshold tau_prime
env.consts.tau_prime = env.opts.tau * (env.consts.w2'*(env.consts.y.^2));

% check for valid C object and extract partitioning
if ~isa(env.opts.C, 'linops.Blockwise')
    error('ffd:input:C:type','C must be a linops.Blockwise object');
end

if size(env.opts.C, 1) ~= env.consts.M
    error('ffd:input:C:rows','C must have a total of %d rows', env.consts.M);
end

if numel(env.opts.C.rowSplits) ~= numel(env.opts.C.colSplits)
    error('ffd:input:C:diagonal','C must be block diagonal');
end

if size(env.opts.C,2) ~= size(env.consts.KH,1) || ...
        ~isequal(env.opts.C.colSplits(:),env.consts.KH.rowSplits(:))
    error('ffd:input:C:compatible',...
          'C''s column partitioning must be identical to K''s row partitioning');
end

env.consts.yBlocks = env.opts.C.rowBlocks;
env.consts.yIdx1 = env.opts.C.rowFirst(1:env.consts.yBlocks);
env.consts.yIdx2 = env.opts.C.rowLast(1:env.consts.yBlocks);

env.consts.ypBlocks = env.opts.C.colBlocks;
env.consts.ypIdx1 = env.opts.C.colFirst(1:env.consts.ypBlocks);
env.consts.ypIdx2 = env.opts.C.colLast(1:env.consts.ypBlocks);

% check for valid Q object and extract partitioning
if isempty(env.opts.Q)
    env.consts.xxBlocks = 0;
    env.consts.xxIdx1 = [];
    env.consts.xxIdx2 = [];
    env.consts.QxxBlocks = 0;
    env.consts.QxxIdx1 = [];
    env.consts.QxxIdx2 = [];
elseif isa(env.opts.Q, 'linops.Blockwise')
    if size(env.opts.Q,2) ~= env.consts.N
        error('ffd:input:Q:cols','Q must have %d columns', env.consts.N);
    end
    env.consts.xxBlocks = env.opts.Q.colBlocks;
    env.consts.xxIdx1 = env.opts.Q.colFirst(1:env.consts.xxBlocks);
    env.consts.xxIdx2 = env.opts.Q.colLast(1:env.consts.xxBlocks);
    env.consts.QxxBlocks = env.opts.Q.rowBlocks;
    env.consts.QxxIdx1 = env.opts.Q.rowFirst(1:env.consts.QxxBlocks);
    env.consts.QxxIdx2 = env.opts.Q.rowLast(1:env.consts.QxxBlocks);
else
    error('ffd:input:Q:type','Q must either be empty or a linops.Blockwise object');
end

env.consts.compareX = false;
env.consts.compareJ = false;

% Check for valid Xthe or Jthe
if ~isempty(env.opts.Xthe)
    if isequal(size(env.opts.Xthe),env.consts.Xsize)
        env.consts.compareX = true;
    else
        error('ffd:input:Xthe', 'invalid Xthe size');
    end
elseif ~isempty(env.opts.Jthe)
    if isequal(size(env.opts.Jthe),[env.consts.N, env.consts.N])
        env.consts.compareJ = true;
    else
        error('ffd:input:Jthe', 'invalid Jthe size');
    end
end

% Setup additional output parameters
if nargout > 1
    iterations = struct;
    iterations.fvals = zeros(env.opts.imax,1);
    iterations.yerrs = zeros(env.opts.imax,1);
    iterations.G2s = zeros(env.opts.imax,1);
    iterations.S2s = zeros(env.opts.imax,1);
    iterations.ts = zeros(env.opts.imax,1);
    if env.consts.compareJ || env.consts.compareX
        iterations.Jerrs = zeros(env.opts.imax,1);
    end
end

% the current state of the algorithm (varies per iteration)
env.state = struct;

% Generate initial guess
if isempty(env.opts.X1)
    env.state.X = zeros(env.consts.Xsize);
else
    if ~isequal(size(env.opts.X1),env.consts.Xsize)
        error('ffd:input:X1', 'X1 needs to be of size %s%d', sprintf('%d-by-',env.consts.Xsize(1:end-1)), env.consts.Xsize(end));
    end
    if env.opts.verbose
        if env.opts.X1seed
            fprintf('Using given initial hint for X...\n');
        else
            fprintf('Using given initial value for X...\n');
        end
    end
    env.state.X = env.opts.X1;
end

% fill in rest of state with initial values
env.state.iteration = 0;
env.state.Delta = [];
env.state.G = zeros(env.consts.Xsize);
env.state.Ghat = zeros(env.consts.Xsize);
env.state.fval = 0;
env.state.fval_pre = 0;
env.state.yerr = 0;
env.state.S = zeros(env.consts.Xsize);
env.state.reset_cg = false;
env.state.terminate = false;
env.state.G_previous = zeros(env.consts.Xsize);
env.state.Ghat_previous = zeros(env.consts.Xsize);
env.state.quartic = zeros(1,5);
env.state.cubic = zeros(1,4);
env.state.fval_next = 0;
env.state.decrement = 0;
env.state.alpha = 0;
env.state.beta = 0;
env.state.t = 0;

if env.consts.compareX
    env.state.Jerr = ffd.factored_distance(env.opts.Xthe(env.opts.maskJ,:), env.state.X(env.opts.maskJ,:))/sum(env.opts.maskJ);
elseif env.consts.compareJ
    % set initial value of J for error comparison
    env.state.maskedJ = env.state.X(env.opts.maskJ,:)*env.state.X(env.opts.maskJ,:)';
    env.state.Jerr = norm(env.opts.Jthe(env.opts.maskJ,env.opts.maskJ)-env.state.maskedJ,'fro')/sum(env.opts.maskJ);
end

% do initial callback
env.opts.callback(env);

start_time = tic;

for i=1:env.opts.imax

    env.state.iteration = i;

    % calculate the descent direction
    % writes to:
    %    env.state.G (steepest descent direction)
    %    env.state.Ghat (descent direction)
    %    env.state.reset_cg (set to true to force a cg reset)
    %    env.state.fval (value of merit function)
    %    env.state.yerr (rmse of measurements not masked out by yerr_mask)
    %    env.state.fval_pre (quartic part of merit function not masked out)
    %    env.state.Delta (optional, the error per measurement)
    %    env.state.terminate (set to true to stop iterating)
    if env.state.iteration==1 && (env.opts.X1seed || ~any(env.state.X(:)))
        initial = ffd.descent.Initial;
        initial(env);
    else
        env.opts.descent(env);
    end

    if env.state.terminate
        % terminate and truncate per-iteration values
        if exist('iterations','var')
            iterations.fvals = iterations.fvals(1:i-1);
            iterations.yerrs = iterations.yerrs(1:i-1);
            iterations.G2s = iterations.G2s(1:i-1);
            iterations.S2s = iterations.S2s(1:i-1);
            iterations.ts = iterations.ts(1:i-1);
            if env.consts.compareJ || env.consts.compareX
                iterations.Jerrs = iterations.Jerrs(1:i-1);
            end
        end
        break;
    end

    if env.consts.compareX
        env.state.Jerr = ffd.factored_distance(env.opts.Xthe(env.opts.maskJ,:), env.state.X(env.opts.maskJ,:))/sum(env.opts.maskJ);
    elseif env.consts.compareJ
        env.state.Jerr = norm(env.opts.Jthe(env.opts.maskJ,env.opts.maskJ)-env.state.maskedJ,'fro')/sum(env.opts.maskJ);
    end
    
    % compute conjugate gradient
    if env.opts.cg && ~env.state.reset_cg
        % compute conjugate gradient using preconditioned modified 
        % Polak-Ribiere method
        r0 = env.state.G(:);
        r1 = env.state.G_previous(:);
        d0 = env.state.Ghat(:);
        d1 = env.state.Ghat_previous(:);
        denom = real(r1'*d1);
        if denom <= 0
            env.state.beta = 0;
        else
            env.state.beta = max(0, (real(r0'*d0)-real(r0'*d1)) / denom);
        end
        env.state.S = env.state.Ghat + env.state.beta*env.state.S;
        env.state.G_previous = env.state.G;
        env.state.Ghat_previous = env.state.Ghat;
        clear r0 r1 d0 d1
    else
        % no conjugate gradient or during a reset
        env.state.beta = 0;
        env.state.S = env.state.Ghat;
        env.state.G_previous = env.state.G;
        env.state.Ghat_previous = env.state.Ghat;
        env.state.reset_cg = false;
    end
    
    % compute the single-variable quartic corresponding to the merit
    % function along the search direction S
    env.state.quartic = [0 0 0 0 0];
    for yIdx=1:env.consts.yBlocks
        KHX_block = env.consts.KH.forward(yIdx,1,env.state.X(env.consts.xIdx1(1):env.consts.xIdx2(1),:));
        KHS_block = env.consts.KH.forward(yIdx,1,env.state.S(env.consts.xIdx1(1):env.consts.xIdx2(1),:));
        for xIdx=2:env.consts.xBlocks
            KHX_block = KHX_block + env.consts.KH.forward(yIdx,xIdx,env.state.X(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
            KHS_block = KHS_block + env.consts.KH.forward(yIdx,xIdx,env.state.S(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
        end
        if ~isempty(env.state.Delta)
            delta_block = env.state.Delta(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx));
        else
            delta_block = env.consts.y(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx)) ...
                - env.opts.C.forward(yIdx,yIdx,sum(real(KHX_block).^2,2)+sum(imag(KHX_block).^2,2));
        end
        b_block = -2*env.opts.C.forward(yIdx,yIdx,sum(real(KHX_block).*real(KHS_block),2)+sum(imag(KHX_block).*imag(KHS_block),2));
        c_block = -env.opts.C.forward(yIdx,yIdx,sum(real(KHS_block).^2,2)+sum(imag(KHS_block).^2,2));
        w2_block = env.consts.w2(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx));
        env.state.quartic = env.state.quartic + w2_block'*[c_block.^2,2*c_block.*b_block,2*c_block.*delta_block+b_block.^2, 2*b_block.*delta_block, delta_block.^2];
    end
    clear KHX_block;
    clear KHS_block;            
    clear b_block;
    clear c_block;
    clear delta_block;
    clear w2_block;
    
    % add in energy minimization regularizer to the state.quartic
    for QxxIdx=1:env.consts.QxxBlocks
        temp1 = env.opts.Q.forward(QxxIdx,1,env.state.S(env.consts.xxIdx1(1):env.consts.xxIdx2(1),:));
        temp2 = env.opts.Q.forward(QxxIdx,1,env.state.X(env.consts.xxIdx1(1):env.consts.xxIdx2(1),:));
        for xxIdx=2:env.consts.xxBlocks
            temp1 = temp1 + env.opts.Q.forward(QxxIdx,xxIdx,env.state.S(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
            temp2 = temp2 + env.opts.Q.forward(QxxIdx,xxIdx,env.state.X(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
        end
        quadratic = [temp1(:)'*temp1(:), 2*real(temp1(:)'*temp2(:)), temp2(:)'*temp2(:)];
        env.state.quartic(3:5) = env.state.quartic(3:5) + quadratic;
    end
    clear temp1;
    clear temp2;

    % find the roots of the derivative
    env.state.cubic = [4 3 2 1].*env.state.quartic(1:4);
    alphas = roots(env.state.cubic); 
    alphas = [alphas(imag(alphas)==0); 0.0]; % selecting real roots only, so imaginary roots don't actually get a lower value
    [env.state.fval_next, index] = min(polyval(env.state.quartic,alphas));
    env.state.decrement = env.state.fval - env.state.fval_next;
    env.state.alpha = alphas(index);
    env.state.t = toc(start_time);
    
    % write per-iteration values
    if exist('iterations','var')
        iterations.fvals(i) = env.state.fval;
        iterations.yerrs(i) = env.state.yerr;
        iterations.G2s(i) = env.state.G(:)'*env.state.G(:);
        iterations.S2s(i) = env.state.S(:)'*env.state.S(:);
        iterations.ts(i) = env.state.t;
        if env.consts.compareJ || env.consts.compareX
            iterations.Jerrs(i) = env.state.Jerr;
        end
    end
    
    % do callback
    env.opts.callback(env);
    % update
    env.state.X = env.state.X+env.state.alpha*env.state.S;
    if env.consts.compareJ
        env.state.maskedJ = env.state.X(env.opts.maskJ,:)*env.state.X(env.opts.maskJ,:)';
    end
end % end iterations

X = env.state.X;

end % end function

