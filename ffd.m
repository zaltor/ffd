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
%      FFD(..., 'callback', CALLBACK, ...) will invoke CALLBACK(OPTS,
%      STATE) before the first iteration and after each iteration.
%
%      FFD(..., 'cg', false, ...) will disable conjugate gradient and cause
%      the step direction to be simply the (preconditioned) gradient.
%
%      FFD(..., 'precond', 'none', ...) will force FFD to not use
%      preconditioning
%
%      FFD(..., 'precond', 'whiten', ...) will use preconditioning to
%      reduce the negative effects of factorization. This is the default
%      behavior.
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

% check for valid y and extract size
if size(y,1) ~= numel(y)
    error('ffd:input:y','y must be an M-by-1 vector');
end
M = size(y,1);

% check for valid KH and extract size and partitioning
if ~isa(KH, 'linops.Blockwise')
    error('ffd:input:KH','KH must be a linops.Blockwise object');
end
N = size(KH,2);
xBlocks = KH.colBlocks;
xIdx1 = KH.colFirst(1:xBlocks);
xIdx2 = KH.colLast(1:xBlocks);

% get options
opts = struct;
opts.L = 1000;
opts.w = [];
opts.X0 = 'white';
opts.R = N;
opts.Jthe = [];
opts.Xthe = [];
opts.verbose = true;
opts.callback = ffd.callbacks.Status(1);
opts.precond = 'whiten';
opts.cg = true;
opts.rs = RandStream.getGlobalStream;
opts.C = linops.Identity(size(KH,1),KH.rowSplits);
opts.Q = [];
opts.yerr_mask = true(M,1);
opts.Jerr_mask = true(N,1);

opts = mhelpers.getopts(opts, varargin{:});

% size of X
Xsize = [N, opts.R];

% check for valid C object and extract paritioning
if ~isa(opts.C, 'linops.Blockwise')
    error('ffd:input:C:type','C must be a linops.Blockwise object');
end

if size(opts.C, 1) ~= M
    error('ffd:input:C:rows','C must have a total of %d rows', M);
end

if numel(opts.C.rowSplits) ~= numel(opts.C.colSplits)
    error('ffd:input:C:diagonal','C must be block diagonal');
end

if size(opts.C,2) ~= size(KH,1) || ~isequal(opts.C.colSplits(:),KH.rowSplits(:))
    error('ffd:input:C:compatible',...
          'C''s column partitioning must be identical to K''s row partitioning');
end

yBlocks = opts.C.rowBlocks;
yIdx1 = opts.C.rowFirst(1:yBlocks);
yIdx2 = opts.C.rowLast(1:yBlocks);

% check for valid Q object and extract partitioning
if isempty(opts.Q)
    opts.Q = linops.Matrix(sparse(0,N));
end
    
if ~isa(opts.Q, 'linops.Blockwise')
    error('ffd:input:Q:type','Q must either be empty or a linops.Blockwise object');
end

if size(opts.Q,2) ~= N
    error('ffd:input:Q:cols','Q must have %d columns', N);
end

xxBlocks = opts.Q.colBlocks;
xxIdx1 = opts.Q.colFirst(1:xxBlocks);
xxIdx2 = opts.Q.colLast(1:xxBlocks);
QxxBlocks = opts.Q.rowBlocks;

% precalculate w^2
if isempty(opts.w)
    w2 = true(M,1);
else
    w2 = opts.w.*conj(opts.w);
end

compareX = false;
compareJ = false;

% Check for valid Xthe or Jthe
if ~isempty(opts.Xthe)
    if isequal(size(opts.Xthe),Xsize)
        compareX = true;
    else
        error('ffd:input:Xthe', 'invalid Xthe size');
    end
elseif ~isempty(opts.Jthe)
    if isequal(size(opts.Jthe),[N, N])
        compareJ = true;
    else
        error('ffd:input:Jthe', 'invalid Jthe size');
    end
end

% Setup additional output parameters
if nargout > 1
    iterations = struct;
    iterations.fvals = zeros(opts.L,1);
    iterations.yerrs = zeros(opts.L,1);
    iterations.G2s = zeros(opts.L,1);
    iterations.S2s = zeros(opts.L,1);
    iterations.ts = zeros(opts.L,1);
    if compareJ || compareX
        iterations.Jerrs = zeros(opts.L,1);
    end
end

% the current state of the algorithm
state = struct;

% Generate initial guess
if strcmp(opts.X0,'white') == true
    if opts.verbose
        fprintf('Generating random initial value for X...\n');
    end
    state.X = opts.rs.randn(Xsize) + 1j*opts.rs.randn(Xsize);
    % forward propagate X and rescale X so that total intensity == y's
    total_intensity = 0;
    for yIdx = 1:yBlocks
        for xIdx = 1:xBlocks
            KHX_block = KH.forward(yIdx,xIdx,state.X(xIdx1(xIdx):xIdx2(xIdx),:));
            total_intensity = total_intensity + sum(opts.C.forward(yIdx,yIdx,sum(KHX_block.*conj(KHX_block),2)));
        end
    end
    clear KHX_block;
    scaling = sqrt(sum(y)/total_intensity);
    state.X = scaling*state.X;
elseif strcmp(opts.X0,'guess') == true
    if opts.verbose
        fprintf('Generating random guess for X...\n');
    end
    state.X = zeros(Xsize);
    for yIdx = 1:yBlocks
        for xIdx = 1:xBlocks
            ytemp = repmat(sqrt(y(yIdx1(yIdx):yIdx2(yIdx))),[1 Xsize(2)]) .* ...
                    exp(2j*pi*opts.rs.rand([yIdx2(yIdx)-yIdx1(yIdx)+1,Xsize(2)]));
            ytemp = opts.C.adjoint(yIdx,yIdx,ytemp);
            state.X(xIdx1(xIdx):xIdx2(xIdx),:) = ...
                state.X(xIdx1(xIdx):xIdx2(xIdx),:) + KH.adjoint(yIdx,xIdx,ytemp);
            clear ytemp;
        end
    end
    % forward propagate X and rescale X so that total intensity == y's
    total_intensity = 0;
    for yIdx = 1:yBlocks
        for xIdx = 1:xBlocks
            KHX_block = KH.forward(yIdx,xIdx,state.X(xIdx1(xIdx):xIdx2(xIdx),:));
            total_intensity = total_intensity + sum(opts.C.forward(yIdx,yIdx,sum(KHX_block.*conj(KHX_block),2)));
        end
    end
    clear KHX_block;
    scaling = sqrt(sum(y)/total_intensity);
    state.X = scaling*state.X;
else
    if ~isequal(size(opts.X0),Xsize)
        error('ffd:input:X0', 'X0 needs to be of size %s%d', sprintf('%d-by-',Xsize(1:end-1)), Xsize(end));
    end
    if opts.verbose
        fprintf('Using given initial value for X...\n');
    end
    state.X = opts.X0;
end

% fill in rest of state with initial values
state.iteration = 0;
state.precondX = state.X;
state.G = zeros(Xsize);
state.Ghat = zeros(Xsize);
state.fval = 0;
state.fval_pre = 0;
state.yerr = 0;
state.S = zeros(Xsize);
state.G_previous = zeros(Xsize);
state.Ghat_previous = zeros(Xsize);
state.quartic = zeros(1,5);
state.cubic = zeros(1,4);
state.fval_next = 0;
state.decrement = 0;
state.alpha = 0;
state.beta = 0;
state.t = 0;

if compareX
    state.Jerr = ffd.factored_distance(opts.Xthe, state.X)/N;
elseif compareJ
    % set initial value of J for error comparison
    state.J = state.X*state.X';
    state.Jerr = norm(opts.Jthe-state.J,'fro')/N;
end

% do initial callback
opts.callback(opts,state);

start_time = tic;

for i=1:opts.L
    state.iteration = i;
    % preconditioning to ameliorate distortion effects caused by
    % factorization
    switch(opts.precond)
        case 'none'
            state.precondX = state.X;
        case 'whiten'
            [U_,S_,V_] = svd(state.X, 'econ');
            if S_(1,1) == 0
                % X is the zero matrix, so make up something
                if Xsize(1) > Xsize(2)
                    X_fake = repmat(eye(Xsize(2)),[ceil(Xsize(1)/Xsize(2)),1]);
                else
                    X_fake = repmat(eye(Xsize(1)),[1,ceil(Xsize(2)/Xsize(1))]);
                end
                state.precondX = X_fake(1:Xsize(1),1:Xsize(2));
                clear X_fake;
            else
                new_S = sqrt(trace(S_.^2)/size(S_,1))*eye(size(S_,1));
                state.precondX = U_*new_S*V_';
                clear new_S;
            end
            clear U_;
            clear S_;
            clear V_;
    end
    
    % compute error in intensity and steepest descent direction as well as
    % preconditioned steepest descent direction
    state.G = zeros(Xsize);
    state.Ghat = zeros(Xsize);
    state.fval = 0;
    state.yerr = 0;
    state.fval_pre = 0;
    for yIdx=1:yBlocks
        w2_block = w2(yIdx1(yIdx):yIdx2(yIdx));
        KHX_block = 0;
        for xIdx = 1:xBlocks
            KHX_block = KHX_block + KH.forward(yIdx,xIdx,state.X(xIdx1(xIdx):xIdx2(xIdx),:));
        end
        y_block = opts.C.forward(yIdx,yIdx,sum(KHX_block.*conj(KHX_block),2));
        delta_block = y(yIdx1(yIdx):yIdx2(yIdx)) - y_block;
        mask_block = opts.yerr_mask(yIdx1(yIdx):yIdx2(yIdx));
        E = diag(sparse(opts.C.adjoint(yIdx,yIdx,delta_block.*w2_block)));
        fval_inc = w2_block'*(delta_block.*conj(delta_block));
        state.fval = state.fval + fval_inc;
        state.fval_pre = state.fval_pre + w2_block'*(delta_block.*mask_block.*conj(delta_block));
        state.yerr = state.yerr + mask_block'*(delta_block.*conj(delta_block));
        clear delta_block;
        for xIdx = 1:xBlocks
            state.G(xIdx1(xIdx):xIdx2(xIdx),:) = ...
                state.G(xIdx1(xIdx):xIdx2(xIdx),:) + ...
                    4*KH.adjoint(yIdx,xIdx,E*KHX_block);
        end
        clear KHX_block;
        switch(opts.precond)
            case 'none'
                state.Ghat = state.G;
            case 'whiten'
                KHprecondX_block = 0;
                for xIdx = 1:xBlocks
                    KHprecondX_block = KHprecondX_block + KH.forward(yIdx,xIdx,state.precondX(xIdx1(xIdx):xIdx2(xIdx),:));
                    state.Ghat(xIdx1(xIdx):xIdx2(xIdx),:) = ...
                        state.Ghat(xIdx1(xIdx):xIdx2(xIdx),:) + ...
                            4*KH.adjoint(yIdx,xIdx,E*KHprecondX_block);
                end
                clear KHprecondX_block;
        end
    end
    
    for QxxIdx=1:QxxBlocks
        Qxx_block = 0;
        for xxIdx=1:xxBlocks
            Qxx_block = Qxx_block + ...
                opts.Q.forward(QxxIdx,xxIdx,state.X(xxIdx1(xxIdx):xxIdx2(xxIdx),:));
        end
        state.fval = state.fval + Qxx_block(:)'*Qxx_block(:);
        for xxIdx=1:xxBlocks
            state.G(xxIdx1(xxIdx):xxIdx2(xxIdx),:) = ...
                state.G(xxIdx1(xxIdx):xxIdx2(xxIdx),:) - ...
                    2*opts.Q.adjoint(QxxIdx,xxIdx,Qxx_block);
        end
        clear Qxx_block;
        switch(opts.precond)
            case 'none'
                state.Ghat = state.G;   
            case 'whiten'
                Qxx_block = 0;
                for xxIdx=1:xxBlocks
                    Qxx_block = Qxx_block + ...
                        opts.Q.forward(QxxIdx,xxIdx,state.precondX(xxIdx1(xxIdx):xxIdx2(xxIdx),:));
                end
                for xxIdx=1:xxBlocks
                    state.Ghat(xxIdx1(xxIdx):xxIdx2(xxIdx),:) = ...
                        state.Ghat(xxIdx1(xxIdx):xxIdx2(xxIdx),:) - ...
                            2*opts.Q.adjoint(QxxIdx,xxIdx,Qxx_block);
                end
                clear Qxx_block;
        end
    end

    % convert total squared error into rms error
    state.yerr = sqrt(state.yerr/sum(opts.yerr_mask));

    % compute Jerr
    if compareX
        state.Jerr = ffd.factored_distance(opts.Xthe, state.X)/N;
    elseif compareJ
        state.Jerr = norm(opts.Jthe-J,'fro')/N;
    end
    
    % compute conjugate gradient
    if i == 1 || ~opts.cg
        % first iteration, no conjugate gradient
        state.S = state.Ghat;
        state.G_previous = state.G;
        state.Ghat_previous = state.Ghat;
    else
        % compute conjugate gradient using preconditioned modified 
        % Polak-Ribiere method
        r0 = state.G(:);
        r1 = state.G_previous(:);
        d0 = state.Ghat(:);
        d1 = state.Ghat_previous(:);
        state.beta = max(0, real((r0'*(d0-d1))/(r1'*d1)));
        state.S = state.Ghat + state.beta*state.S;
        state.G_previous = state.G;
        state.Ghat_previous = state.Ghat;
    end
        
    % compute the single-variable state.quartic corresponding to the merit
    % function along the search direction S
    state.quartic = [0 0 0 0 0];
    for yIdx=1:yBlocks
        KHX_block = 0;
        KHS_block = 0;
        for xIdx=1:xBlocks
            KHX_block = KHX_block + KH.forward(yIdx,xIdx,state.X(xIdx1(xIdx):xIdx2(xIdx),:));
            KHS_block = KHS_block + KH.forward(yIdx,xIdx,state.S(xIdx1(xIdx):xIdx2(xIdx),:));
        end
        y_block = opts.C.forward(yIdx,yIdx,sum(KHX_block.*conj(KHX_block),2));
        delta_block = y(yIdx1(yIdx):yIdx2(yIdx)) - y_block;
        clear y_block;
        b_block = -opts.C.forward(yIdx,yIdx,sum(2*real(KHX_block.*conj(KHS_block)),2));
        clear KHX_block;
        c_block = -opts.C.forward(yIdx,yIdx,sum(KHS_block.*conj(KHS_block),2));
        clear KHS_block;
        w2_block = w2(yIdx1(yIdx):yIdx2(yIdx));
        state.quartic = state.quartic + w2_block'*[c_block.^2,2*c_block.*b_block,2*c_block.*delta_block+b_block.^2, 2*b_block.*delta_block, delta_block.^2];
        clear b_block;
        clear c_block;
        clear delta_block;
        clear w2_block;
    end
    
    % add in energy minimization regularizer to the state.quartic
    for QxxIdx=1:QxxBlocks
        temp1 = 0;
        temp2 = 0;
        for xxIdx=1:xxBlocks
            temp1 = temp1 + opts.Q.forward(QxxIdx,xxIdx,state.S(xxIdx1(xxIdx):xxIdx2(xxIdx),:));
            temp2 = temp2 + opts.Q.forward(QxxIdx,xxIdx,state.X(xxIdx1(xxIdx):xxIdx2(xxIdx),:));
        end
        quadratic = [temp1(:)'*temp1(:), 2*real(temp1(:)'*temp2(:)), temp2(:)'*temp2(:)];
        clear temp1;
        clear temp2;
        state.quartic(3:5) = state.quartic(3:5) + quadratic;
    end

    % find the roots of the derivative
    state.cubic = [4 3 2 1].*state.quartic(1:4);
    alphas = roots(state.cubic); 
    alphas = [alphas(imag(alphas)==0); 0.0]; % selecting real roots only, so imaginary roots don't actually get a lower value
    [state.fval_next, index] = min(polyval(state.quartic,alphas));
    state.decrement = state.fval - state.fval_next;
    state.alpha = alphas(index);
    state.t = toc(start_time);
    
    % write per-iteration values
    if exist('iterations','var')
        iterations.fvals(i) = state.fval;
        iterations.yerrs(i) = state.yerr;
        iterations.G2s(i) = state.G(:)'*state.G(:);
        iterations.S2s(i) = state.S(:)'*state.S(:);
        iterations.ts(i) = state.t;
        if compareJ || compareX
            iterations.Jerrs(i) = state.Jerr;
        end
    end
    
    % do callback
    opts.callback(opts,state);
    
    % update
    state.X = state.X+state.alpha*state.S;
    if compareJ
        state.J = state.X*state.X';
    end
end % end iterations

X = state.X;

end % end function

