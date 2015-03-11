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
%   mode, and Y is an M-by-1 nonnegative vector containing the
%   intensity measurements.
%
%   X = FFD(Y, KH, 'w', W) will instead minimize the weighted least squared
%   error in intensity measurements, with weighting vector W, which is
%   expected to be the same size as Y. Each element of W weighs the
%   absolute value of the error for the same index element in Y.
%
%   X = FFD(..., 'C', C) will minimize the (weighted) least squared error
%   in linear combinations of intensity measurements, where C is a
%   linops.Blockwise object. That is, if Z is a vector of the intensity
%   values derived from the current iterate, then the merit function is
%   equal to norm(Y-C*Z,2)^2. Note: C must be block-wise diagonal and the 
%   row partitioning pattern for KH must be the same as the row and column 
%   partitioning patterns for C.
%
%   X = FFD(Y, KH, 'R', R) performs coherence retrieval where the number
%   of modes is limited to R. In this case, X will be an N-by-R matrix. For
%   phase retrieval (i.e. a fully coherent field) R should be set to 1.
%   
%   [X, ITERATIONS] = FFD(...) stores per-iteration numbers such as the
%   merit function value, the RMS error in the intensity, etc. in the
%   structure ITERATIONS.
%
%   [X, ITERATIONS] = FFD(Y, KH, 'Jthe', J0, ...) will additionally store
%   in ITERATIONS the per-iteration RMS error in the mutual intensity with
%   respect to J0.
%
%   [X, ITERATIONS] = FFD(Y, KH, 'Xthe', X0, ...) will additionally store
%   in ITERATIONS the per-iteration RMS error in the mutual intensity with
%   respect to X0*X0'. This is more efficient than the previous approach if
%   R is set to 1.
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
%      FFD(..., 'L', L, ...) will cause FFD to run for L iterations. The
%      default value is 1000.
%
%      FFD(..., 'w', W, ...) will use the weights in W for weighted least
%      squares. W must be the same size as Y. The default value is a vector
%      of ones.
%
%      FFD(..., 'R', R, ...) will restrict the solution to R modes. In
%      practice, setting R to be less than N will cause FFD to converge
%      poorly without enough measurements. Convergence guarantees are still
%      an open topic of research for this situation. The default value is
%      N.
%
%      FFD(..., 'X0', 'white', ...) will use an initial value equal to 
%      white noise scaled such that the total output intensity when
%      propagated is equal to the sum of the measurements. This is the
%      default behavior.
%
%      FFD(..., 'X0', 'guess', ...) will use the adjoint operator on random
%      phases and backpropagate to the source, rescaling to match the sum
%      of output intensities with the sum of the measurements.
%
%      FFD(..., 'X0', X0, ...) will use the specific value X0 as the
%      initial value.
%
%      FFD(..., 'Jthe', J_THE, ...) will effectively compute 
%      norm(X*X'-J_THE, 'fro')/N at each iteration, storing into JERRS. 
%
%      FFD(..., 'Xthe', X_THE, ...) will effectively compute
%      norm(X*X'-X_THE*X_THE', 'fro')/N at each iteration, storing into
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
%      FFD(..., 'cg', false, ...) will disable conjugate gradient and cause
%      the step direction to be simply the (preconditioned) gradient.
%
%      FFD(..., 'precond', ffd.precond.None, ...) will force FFD to not use
%      preconditioning
%
%      FFD(..., 'precond', ffd.precond.Equalize, ...) will use mode energy
%      equalization-based preconditioning to reduce the negative effects of
%      factorization.  This is the default behavior.
%
%      FFD(..., 'Q', Q, ...) will add an energy minimization term
%      trace((Q*X)*(Q*X)') to the merit function, where Q is a
%      linops.Blockwise object.
%
%      FFD(..., 'rs', RS, ...) will force FFD to use a specific RandStream
%      for generating random numbers (i.e. for setting the initial value).
%      If unspecified, FFD will use the global random number generator.
%
%      FFD(..., 'Jerr_mask', JERR_MASK, ...) will force FFD to only compute
%      JERRS based on certain elements of J. JERR_MASK is a logical N-by-1
%      array (i.e. same number of rows as X and J_THE), and an element is
%      true if the corresponding row in X is to be used for mutual
%      intensity RMS error calculation.
%
%      FFD(..., 'yerr_mask', YERR_MASK, ...) will force FFD to only compute
%      YERRS based on certain elements of Y. YERR_MASK is a logical M-by-1
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
env.opts.L = 1000;
env.opts.w = [];
env.opts.X0 = 'white';
env.opts.R = env.consts.N;
env.opts.Jthe = [];
env.opts.Xthe = [];
env.opts.verbose = true;
env.opts.callback = ffd.callbacks.Status(1);
env.opts.precond = ffd.precond.Equalize;
env.opts.cg = true;
env.opts.rs = RandStream.getGlobalStream;
env.opts.C = linops.Identity(size(env.consts.KH,1),env.consts.KH.rowSplits);
env.opts.Q = [];
env.opts.yerr_mask = true(env.consts.M,1);
env.opts.Jerr_mask = true(env.consts.N,1);

env.opts = mhelpers.getopts(env.opts, varargin{:});

% size of X
env.consts.Xsize = [env.consts.N, env.opts.R];

% precalculate w^2
if isempty(env.opts.w)
    env.consts.w2 = true(env.consts.M,1);
else
    env.consts.w2 = env.opts.w.*conj(env.opts.w);
end

% check for valid C object and extract paritioning
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

% check for valid Q object and extract partitioning
if isempty(env.opts.Q)
    env.opts.Q = linops.Matrix(sparse(0,env.consts.N));
end
    
if ~isa(env.opts.Q, 'linops.Blockwise')
    error('ffd:input:Q:type','Q must either be empty or a linops.Blockwise object');
end

if size(env.opts.Q,2) ~= env.consts.N
    error('ffd:input:Q:cols','Q must have %d columns', env.consts.N);
end

env.consts.xxBlocks = env.opts.Q.colBlocks;
env.consts.xxIdx1 = env.opts.Q.colFirst(1:env.consts.xxBlocks);
env.consts.xxIdx2 = env.opts.Q.colLast(1:env.consts.xxBlocks);
env.consts.QxxBlocks = env.opts.Q.rowBlocks;

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
    iterations.fvals = zeros(env.opts.L,1);
    iterations.yerrs = zeros(env.opts.L,1);
    iterations.G2s = zeros(env.opts.L,1);
    iterations.S2s = zeros(env.opts.L,1);
    iterations.ts = zeros(env.opts.L,1);
    if env.consts.compareJ || env.consts.compareX
        iterations.Jerrs = zeros(env.opts.L,1);
    end
end

% the current state of the algorithm (varies per iteration)
env.state = struct;

% Generate initial guess
if strcmp(env.opts.X0,'white') == true
    if env.opts.verbose
        fprintf('Generating random initial value for X...\n');
    end
    env.state.X = env.opts.rs.randn(env.consts.Xsize) + 1j*env.opts.rs.randn(env.consts.Xsize);
    % forward propagate X and rescale X so that total intensity == y's
    total_intensity = 0;
    for yIdx = 1:env.consts.yBlocks
        for xIdx = 1:env.consts.xBlocks
            KHX_block = env.consts.KH.forward(yIdx,xIdx,env.state.X(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
            total_intensity = total_intensity + sum(env.opts.C.forward(yIdx,yIdx,sum(KHX_block.*conj(KHX_block),2)));
        end
    end
    clear KHX_block;
    scaling = sqrt(sum(env.consts.y)/total_intensity);
    env.state.X = scaling*env.state.X;
elseif strcmp(env.opts.X0,'guess') == true
    if env.opts.verbose
        fprintf('Generating random guess for X...\n');
    end
    env.state.X = zeros(Xsize);
    for yIdx = 1:yBlocks
        for xIdx = 1:xBlocks
            ytemp = repmat(sqrt(env.consts.y(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx))),[1 env.consts.Xsize(2)]) .* ...
                    exp(2j*pi*env.opts.rs.rand([env.consts.yIdx2(yIdx)-env.consts.yIdx1(yIdx)+1,env.consts.Xsize(2)]));
            ytemp = env.opts.C.adjoint(yIdx,yIdx,ytemp);
            env.state.X(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:) = ...
                env.state.X(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:) + env.consts.KH.adjoint(yIdx,xIdx,ytemp);
            clear ytemp;
        end
    end
    % forward propagate X and rescale X so that total intensity == y's
    total_intensity = 0;
    for yIdx = 1:env.consts.yBlocks
        for xIdx = 1:env.consts.xBlocks
            KHX_block = env.consts.KH.forward(yIdx,xIdx,env.state.X(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
            total_intensity = total_intensity + sum(env.opts.C.forward(yIdx,yIdx,sum(KHX_block.*conj(KHX_block),2)));
        end
    end
    clear KHX_block;
    scaling = sqrt(sum(env.consts.y)/total_intensity);
    env.state.X = scaling*env.state.X;
else
    if ~isequal(size(env.opts.X0),env.consts.Xsize)
        error('ffd:input:X0', 'X0 needs to be of size %s%d', sprintf('%d-by-',env.consts.Xsize(1:end-1)), env.consts.Xsize(end));
    end
    if env.opts.verbose
        fprintf('Using given initial value for X...\n');
    end
    env.state.X = env.opts.X0;
end

% fill in rest of state with initial values
env.state.iteration = 0;
env.state.delta = zeros(env.consts.M,1);
env.state.G = zeros(env.consts.Xsize);
env.state.Ghat = zeros(env.consts.Xsize);
env.state.fval = 0;
env.state.fval_pre = 0;
env.state.yerr = 0;
env.state.S = zeros(env.consts.Xsize);
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
    env.state.Jerr = ffd.factored_distance(env.opts.Xthe, env.state.X)/env.consts.N;
elseif env.consts.compareJ
    % set initial value of J for error comparison
    env.state.J = env.state.X*env.state.X';
    env.state.Jerr = norm(env.opts.Jthe-env.state.J,'fro')/env.consts.N;
end

% do initial callback
env.opts.callback(env);

start_time = tic;

for i=1:env.opts.L

    % initialize
    env.state.iteration = i;
    env.state.G = zeros(env.consts.Xsize);
    env.state.Ghat = zeros(env.consts.Xsize);

    % compute error in intensity and steepest descent direction
    for yIdx=1:env.consts.yBlocks
        w2_block = env.consts.w2(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx));
        KHX_block = 0;
        for xIdx = 1:env.consts.xBlocks
            KHX_block = KHX_block + env.consts.KH.forward(yIdx,xIdx,env.state.X(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
        end
        y_block = env.opts.C.forward(yIdx,yIdx,sum(KHX_block.*conj(KHX_block),2));
        delta_block = env.consts.y(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx)) - y_block;
        env.state.delta(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx)) = delta_block;
        E = diag(sparse(env.opts.C.adjoint(yIdx,yIdx,delta_block.*w2_block)));
        clear delta_block;
        for xIdx = 1:env.consts.xBlocks
            env.state.G(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:) = ...
                env.state.G(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:) + ...
                    4*env.consts.KH.adjoint(yIdx,xIdx,E*KHX_block);
        end
        clear KHX_block;
    end
    
    % compute merit function value and error metrics
    temp = env.state.delta.*conj(env.state.delta);
    env.state.fval = env.consts.w2'*temp;
    env.state.fval_pre = env.consts.w2'*(temp.*env.opts.yerr_mask);
    env.state.yerr = sqrt((env.opts.yerr_mask'*temp)/sum(env.opts.yerr_mask));
    clear temp;
    if env.consts.compareX
        env.state.Jerr = ffd.factored_distance(env.opts.Xthe, env.state.X)/env.consts.N;
    elseif env.consts.compareJ
        env.state.Jerr = norm(env.opts.Jthe-env.state.J,'fro')/env.consts.N;
    end
    
    % add quadratic (regularizer) component
    for QxxIdx=1:env.consts.QxxBlocks
        Qxx_block = 0;
        for xxIdx=1:env.consts.xxBlocks
            Qxx_block = Qxx_block + ...
                env.opts.Q.forward(QxxIdx,xxIdx,env.state.X(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
        end
        env.state.fval = env.state.fval + Qxx_block(:)'*Qxx_block(:);
        for xxIdx=1:env.consts.xxBlocks
            env.state.G(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:) = ...
                env.state.G(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:) - ...
                    2*env.opts.Q.adjoint(QxxIdx,xxIdx,Qxx_block);
        end
        clear Qxx_block;
    end
    
    % preconditioning if needed
    [env.state.Ghat, reset_cg, env.state.Ghat_previous] = env.opts.precond(env);
    
    % compute conjugate gradient
    if i == 1 || ~env.opts.cg || reset_cg
        % first iteration, no conjugate gradient
        env.state.beta = 0;
        env.state.S = env.state.Ghat;
        env.state.G_previous = env.state.G;
        env.state.Ghat_previous = env.state.Ghat;
    else
        % compute conjugate gradient using preconditioned modified 
        % Polak-Ribiere method
        r0 = env.state.G(:);
        r1 = env.state.G_previous(:);
        d0 = env.state.Ghat(:);
        d1 = env.state.Ghat_previous(:);
        env.state.beta = max(0, real((r0'*(d0-d1))/(r1'*d1)));
        env.state.S = env.state.Ghat + env.state.beta*env.state.S;
        env.state.G_previous = env.state.G;
        env.state.Ghat_previous = env.state.Ghat;
    end
        
    % compute the single-variable state.quartic corresponding to the merit
    % function along the search direction S
    env.state.quartic = [0 0 0 0 0];
    for yIdx=1:env.consts.yBlocks
        KHX_block = 0;
        KHS_block = 0;
        for xIdx=1:env.consts.xBlocks
            KHX_block = KHX_block + env.consts.KH.forward(yIdx,xIdx,env.state.X(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
            KHS_block = KHS_block + env.consts.KH.forward(yIdx,xIdx,env.state.S(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
        end
        b_block = -env.opts.C.forward(yIdx,yIdx,sum(2*real(KHX_block.*conj(KHS_block)),2));
        clear KHX_block;
        c_block = -env.opts.C.forward(yIdx,yIdx,sum(KHS_block.*conj(KHS_block),2));
        clear KHS_block;
        delta_block = env.state.delta(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx));
        w2_block = env.consts.w2(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx));
        env.state.quartic = env.state.quartic + w2_block'*[c_block.^2,2*c_block.*b_block,2*c_block.*delta_block+b_block.^2, 2*b_block.*delta_block, delta_block.^2];
        clear b_block;
        clear c_block;
        clear delta_block;
        clear w2_block;
    end
    
    % add in energy minimization regularizer to the state.quartic
    for QxxIdx=1:env.consts.QxxBlocks
        temp1 = 0;
        temp2 = 0;
        for xxIdx=1:env.consts.xxBlocks
            temp1 = temp1 + env.opts.Q.forward(QxxIdx,xxIdx,env.state.S(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
            temp2 = temp2 + env.opts.Q.forward(QxxIdx,xxIdx,env.state.X(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
        end
        quadratic = [temp1(:)'*temp1(:), 2*real(temp1(:)'*temp2(:)), temp2(:)'*temp2(:)];
        clear temp1;
        clear temp2;
        env.state.quartic(3:5) = env.state.quartic(3:5) + quadratic;
    end

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
        env.state.J = env.state.X*env.state.X';
    end
end % end iterations

X = env.state.X;

end % end function

