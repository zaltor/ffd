function [X, fvals, yerrs, Jerrs] = ffd(y, sys, varargin)
%FFD    Coherence retrieval using factored form descent
%   X = FFD(Y, SYS) computes a set of (generalized) coherence modes from 
%   intensity measurements and a known optical system specification.  The 
%   result X is an N-by-N complex matrix where N is the number of spatial 
%   samples in the source. Each column of X is a (generalized) coherence
%   mode, and Y is an M-by-1 nonnegative vector containing the
%   measurements, and SYS is a class or struct that specifies the optical 
%   system(s). 
%
%   FFD will attempt to minimize the weighted least squared error in the 
%   measurements. If the 'tracereg' option is not specified, the merit 
%   function can be computed in the following way (for more information 
%   about the other parameters, please see the Optimization Options 
%   section below):
%
%      fval = 0;
%      nparts = SYS.parts;
%      for i=1:nparts
%          block = SYS.parts(i);
%          xhat = SYS.forward(X,i);
%          yprime = sum(xhat.*conj(xhat),2);
%          fval=fval+sum(W(block(1):block(2)).^2 .* ...
%                        (Y(block(1):block(2))-SYS.gather(YPRIME,i)).^2)
%      end
%
%   X = FFD(Y, SYS, 'R', R) performs coherence retrieval where the number
%   of modes is limited to R. In this case, X will be an N-by-R matrix. For
%   phase retrieval (i.e. a fully coherent field) R should be set to 1.
%   
%   [X, FVALS, YERRS] = FFD(...) stores the per-iteration value of the 
%   merit function and the RMS error in the intensity in FVALS and YERRS
%   respectively.
%
%   [X, FVALS, YERRS, JERRS] = FFD(Y, SYS, 'Jthe', J0, ...) will store the
%   per-iteration RMS error (relative to J0) in the mutual intensity in 
%   JERRS.
%
%   [X, FVALS, YERRS, JERRS] = FFD(Y, SYS, 'Xthe', X0, ...) will store the
%   per-iteration RMS error (relative to X0*X0') in the mutual intensity in
%   JERRS. This is more efficient than the previous approach if R is set to
%   1.
%
%   System Specification
%   ====================
% 
%   Optical systems corresponding to the measurements in Y are specified 
%   with SYS, which is either an object or struct which provides a set of 
%   required and optional members/fields. For memory efficiency, FFD will
%   break computation blockwise so that computation for different 
%   contiguous blocks of Y will be done serially.
%
%   SYS must implement the following:
%
%      SYS.N should return the number of spatial samples in the input field
%
%      SYS.parts() should return the number of blocks into which operations
%      will be split
%
%      BLOCK = SYS.parts(I) places the index of the first measurement in
%      the Ith block of Y in BLOCK(1) and places the index of the last
%      measurement in BLOCK(2).
%
%      XHAT = SYS.forward(X, I) should perform the forward transform of the
%      optical system for block I. That is, the set of R fields in the
%      N-by-R matrix X are propagated such that XHAT contains the field
%      corresponding to the Ith block of measurements Y. The forward
%      transform must be a linear operator of X; even affine operators will
%      violate FFD's assumptions.
%
%      X = SYS.adjoint(XHAT, I) should perform the adjoint transform for
%      SYS.forward(X, I).
%
%   SYS may optionally provide the following (both must be provided in
%   order for these to be used):
%
%      Y = SYS.gather(YPRIME, I) takes the set of intensities YPRIME 
%      corresponding to the Ith block of measurements and computes the 
%      actual measurements Y. This is used when we wish to solve for the 
%      least squared error in sums of intensity, e.g. to simulate finite 
%      sensor pixel area which accumulates intensities from multiple output 
%      spatial samples. This function must be a linear operator with
%      nonnegative coefficients when expressed in matrix form. 
%
%      YPRIME = SYS.scatter(Y, I) is the adjoint operator for SYS.gather.
%
%   When both of the above are not specified, it is as if
%   SYS.gather(YPRIME, I) returns YPRIME and SYS.scatter(Y, I) returns Y.
%
%   Optimization Options
%   ====================
%
%   FFD allows for many options to be specified at the command-line in the
%   form of pairs of parameter name and value:
%
%   FFD(Y, SYS, 'param1', VALUE1, 'param2', VALUE2, ...)
%
%   Alternatively, a struct with field names equal to the parameter names and field values equal to the
%   parameter values can be passed in as well using the parameter name
%   'opts': (in this situation, any other options are ignored)
%
%   FFD(Y, SYS, 'opts', OPTS)
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
%      FFD(..., 'verbose', 0, ...) will prevent FFD from printing out
%      progress messages.
%
%      FFD(..., 'verbose', 1, ...) will enable progress messages. Progress
%      messages include various residuals, etc. at the current iteration.
%
%      FFD(..., 'verbtime', INT, ...) will make FFD print out a new
%      progress message if at least INT seconds have passed since the last
%      progress message.
%      
%      FFD(..., 'precond', 'none', ...) will force FFD to not use
%      preconditioning
%
%      FFD(..., 'precond', 'whiten', ...) will use preconditioning to
%      reduce the negative effects of factorization. This is the default
%      behavior.
%
%      FFD(..., 'rs', RS, ...) will force FFD to use a specific RandStream
%      for generating random numbers (i.e. for setting the initial value).
%      If unspecified, FFD will use the global random number generator.
%
%      FFD(..., 'tracereg', TRACEREG, ...) will add an energy minimization
%      term trace((Q*X)*(Q*X)') to the merit function. Like the system 
%      specification, computation for this can also be done blockwise.
%      TRACEREG is either a struct or an object providing the following:
%    
%         TRACEREG.parts gives the number of blocks
%        
%         TRACEREG.parts(I) gives the first and last indexes for block I.
%
%         TRACEREG.forward(X, I) computes the Ith block of Q*X.
%
%         TRACEREG.gradient(X, I) computes the Ith block of Q'*Q*X.
%
%         TRACEREG.adjoint (X, I) computes the Ith block of Q'*X. At least
%         one of TRACEREG.gradient or TRACEREG.adjoint must be provided,
%         and if TRACREG.gradient is provided, then TRACEREG.adjoint is 
%         ignored.
%
%      FFD(..., 'yerr_mask', YERR_MASK, ...) will force FFD to only compute
%      YERRS based on certain elements of Y. YERR_MASK is a logical array
%      the same size as Y, and an element is true if the corresponding
%      element in Y is to be used for RMS error calculation.

% extract system and get dimensions
if size(y,1) ~= numel(y)
    error('ffd:input:ysize','y must be an M-by-1 vector');
end
M = size(y,1);

try
    % duck typing way of getting at structure/object components
    sys.adjoint(sys.forward(zeros(sys.N,0),1),1);
    sys.parts();
    sys.N;
catch err
    error('ffd:input:sysspec','invalid sys structure');
end    

% create dummy gather and scatter operations if necessary
try
    sys.gather(sys.scatter(zeros(M,0),1),1);
    sysgather = @(y,i) sys.gather(y,i);
    sysscatter = @(y,i) sys.scatter(y,i);
catch err
    sysgather = @(y,i) y;
    sysscatter = @(y,i) y;
end

KH = sys.forward;
K = sys.adjoint;
N = sys.N;

% get options
opts = struct;
opts.L = 1000;
opts.w = [];
opts.X0 = 'white';
opts.R = N;
opts.Jthe = [];
opts.Xthe = [];
opts.verbose = 1;
opts.verbtime = 10;
opts.precond = 'whiten';
opts.rs = [];
opts.tracereg = struct;
opts.tracereg.forward = @(x,n) 0;
opts.tracereg.gradient = @(x,n) 0;
opts.tracereg.parts = ffd.sys.parts_template([]);
opts.yerr_mask = true(M,1);

opts = getopts(opts, varargin{:});

try
    opts.tracereg.gradient(zeros(sys.N,0),1);
    tracereggradient = @(x,n) opts.tracereg.gradient(x,n);
catch err
    tracereggradient = @(x,n) opts.tracereg.adjoint(opts.tracereg.forward(x,n),n);
end

% Precalculation setup
if isempty(opts.rs)
    rs = RandStream.getGlobalStream;
else
    rs = opts.rs;
end
w2 = opts.w.^2;
showstatus = opts.verbose >= 1;

doJerr = (nargout > 3);

Xsize = [N, opts.R];

if strcmp(opts.X0,'white') == true
    if showstatus
        fprintf('Generating random initial value for X...\n');
    end
    X = rs.randn(Xsize) + 1j*rs.randn(Xsize);
    % forward propagate X and rescale X so that total intensity == y's
    total_intensity = 0;
    for part_number=1:sys.parts()
        KHX_block = KH(X,part_number);
        total_intensity = total_intensity + sum(sysgather(KHX_block(:).*conj(KHX_block(:)),part_number));
    end
    clear KHX_block;
    scaling = sqrt(sum(y)/total_intensity);
    X = scaling*X;
elseif strcmp(opts.X0,'guess') == true
    if showstatus
        fprintf('Generating random guess for X...\n');
    end
    X = 0;
    for part_number=1:sys.parts()
        block = sys.parts(part_number);
        ytemp = repmat(y(block(1):block(2)),[1 Xsize(2)]) .* ...
                exp(2j*pi*rs.rand([block(2)-block(1)+1,Xsize(2)]));
        X = X + K(ytemp,part_number);
        clear ytemp;
    end
    % forward propagate X and rescale X so that total intensity == y's
    total_intensity = 0;
    for part_number=1:sys.parts()
        KHX_block = KH(X,part_number);
        total_intensity = total_intensity + sum(sysgather(KHX_block(:).*conj(KHX_block(:)),part_number));
    end
    clear KHX_block;
    scaling = sqrt(sum(y)/total_intensity);
    X = scaling*X;
else
    if any(size(opts.X0) ~= Xsize)
        error('ffd:input:X0', 'X0 needs to be of size %s%d', sprintf('%d-by-',Xsize(1:end-1)), Xsize(end));
    end
    if showstatus
        fprintf('Using given initial value for X...\n');
    end
    X = opts.X0;
end

% check Xthe and Jthe
if doJerr
    if size(opts.Xthe,1) == N
        themodes = true;
    elseif size(opts.Jthe,1) == N && size(opts.Jthe,2) == N
        themodes = false;
    else
        error('ffd:input:the', 'a valid Xthe or Jthe is required');
    end
end

if doJerr && ~themodes
    % set value of J for error comparison
    J = X*X';
end

% initialize output variables
fvals = zeros(opts.L, 1);
yerrs = zeros(opts.L, 1);
if doJerr 
    Jerrs = zeros(opts.L, 1);
end

for i=1:opts.L
    % determine if we're updating the display
    if i==1
        last_display = tic;
        shownow = true;
    else
        if toc(last_display) > opts.verbtime
            last_display = tic;
            shownow = true;
        else
            shownow = false;
        end
    end
    % display
    if showstatus && shownow
        fprintf('Iteration %d: ', i);
    end

    % compute preconditioning, if needed
    switch(opts.precond)
        case 'none'
            precondX = X;
        case 'whiten'
            [U_,S_,V_] = svd(X, 'econ');
            if S_(1,1) == 0
                % X is the zero matrix, so make up something
                if size(X,1) > size(X,2)
                    X_fake = repmat(eye(size(X,2)),[ceil(size(X,1)/size(X,2)),1]);
                else
                    X_fake = repmat(eye(size(X,1)),[1,ceil(size(X,2)/size(X,1))]);
                end
                precondX = X_fake(1:size(X,1),1:size(X,2));
            else
                new_S = sqrt(trace(S_.^2)/size(S_,1))*eye(size(S_,1));
                precondX = U_*new_S*V_';
            end
    end
    
    % compute error in intensity and steepest descent direction as well as
    % preconditioned steepest descent direction
    G = 0;
    Ghat = 0;
    fval_pre = 0;
    for part_number=1:sys.parts()
        block = sys.parts(part_number);
        KHX_block = KH(X,part_number);
        delta_block = y(block(1):block(2)) - sysgather(sum(KHX_block .* conj(KHX_block), 2),part_number);
        accept_block = opts.yerr_mask(block(1):block(2));
        if isempty(w2)
            E = diag(sparse(sysscatter(delta_block,part_number)));
        else
            E = diag(sparse(sysscatter(delta_block,part_number).*w2(block(1):block(2))));
        end
        if isempty(w2)
            fval_inc = delta_block'*delta_block;
            fvals(i) = fvals(i) + fval_inc;
            if ~all(accept_block)
                fval_pre = fval_pre + delta_block'*(delta_block.*accept_block);
            else
                fval_pre = fval_pre + fval_inc;
            end
        else
            fval_inc = w2(block(1):block(2))'*(delta_block.^2);
            fvals(i) = fvals(i) + fval_inc;
            if ~all(accept_block)
                fval_pre = fval_pre + w2(block(1):block(2))'*(delta_block.^2.*accept_block);
            else
                fval_pre = fval_pre + fval_inc;
            end
        end
        yerrs(i) = yerrs(i) + delta_block'*(delta_block.*accept_block);
        clear delta_block;
        G = G + 4*K(E*KHX_block,part_number);
        switch(opts.precond)
            case 'none'
                Ghat = G;
            case 'whiten'
                Ghat = Ghat + 4*K(E*KH(precondX,part_number),part_number);
        end
        clear E;
        clear KHX_block;
    end
    
    % initiate tracereg for this iteration
    try
        opts.tracereg.update(X);
    catch err
    end
    
    % apply regularization
    for part_number=1:opts.tracereg.parts()
        temp = opts.tracereg.forward(X,part_number);
        fvals(i) = fvals(i) + sum(temp(:).*conj(temp(:)));
        G = G - 2*tracereggradient(X,part_number);
        switch(opts.precond)
            case 'none'
                Ghat = G;
            case 'whiten'
                Ghat = Ghat - 2*tracereggradient(precondX,part_number);
        end
    end
    
    yerrs(i) = sqrt(yerrs(i)/sum(opts.yerr_mask));
    if doJerr
        if themodes
            Jerrs(i) = ffd.factored_distance(opts.Xthe, X)/N;
        else
            Jerrs(i) = norm(opts.Jthe-J,'fro')/N;
        end
    end
    
    if showstatus && shownow
        fprintf('fval=%g(%g) yerr=%g energy=%g', fvals(i), fval_pre, yerrs(i), real(X(:)'*X(:)));
        if doJerr
            fprintf(' Jerr=%g', Jerrs(i));
        end
    end
    
    if showstatus && shownow
        fprintf(' Gmag=%g', norm(G,'fro'));
    end
    
    if showstatus && shownow
        fprintf(' Ghatproj=%g', real(reshape(Ghat,[],1)'*reshape(G,[],1)));
    end
    
    % compute conjugate gradient
    if i == 1
        % first iteration, no conjugate gradient
        S = Ghat;
        G_previous = G;
        Ghat_previous = Ghat;
    else
        % compute conjugate gradient using preconditioned modified 
        % Polak-Ribiere method
        r0 = reshape(G,[],1);
        r1 = reshape(G_previous,[],1);
        d0 = reshape(Ghat,[],1);
        d1 = reshape(Ghat_previous,[],1);
        beta = max(0, real(r0'*(d0-d1))/(r1'*d1));
        if showstatus && shownow
            fprintf(' beta=%g', beta);
        end
        S = Ghat + beta*S;
        G_previous = G;
        Ghat_previous = Ghat;
    end
        
    if showstatus && shownow
        fprintf(' Smag=%g', norm(S,'fro'));
    end
    
    % perform line search by computing a quartic and minimizing it
    quartic = [0 0 0 0 0];
    for part_number=1:sys.parts()
        block = sys.parts(part_number);
        KHX_block = KH(X,part_number);
        KHS_block = KH(S,part_number);
        delta_block = y(block(1):block(2)) - sysgather(sum(KHX_block .* conj(KHX_block), 2),part_number);
        b_block = -sysgather(sum(2*real(KHX_block.*conj(KHS_block)),2),part_number);
        clear KHX_block;
        c_block = -sysgather(sum(KHS_block.*conj(KHS_block),2),part_number);
        clear KHS_block;
        if isempty(w2)
            quartic = quartic + sum([c_block.^2, 2*c_block.*b_block, 2*c_block.*delta_block+b_block.^2, 2*b_block.*delta_block, delta_block.^2],1);
        else
            quartic = quartic + w2(block(1):block(2))'*[c_block.^2, 2*c_block.*b_block, 2*c_block.*delta_block+b_block.^2, 2*b_block.*delta_block, delta_block.^2];
        end
        clear b_block c_block;
    end
    for part_number=1:opts.tracereg.parts()
        % add in regularization quadratic to the quartic
        temp1 = opts.tracereg.forward(S,part_number);
        temp2 = opts.tracereg.forward(X,part_number);
        quadratic = [sum(temp1(:).*conj(temp1(:))), sum(2*real(temp1(:).*conj(temp2(:)))), sum(temp2(:).*conj(temp2(:)))];
        clear temp1;
        clear temp2;
        quartic(3:5) = quartic(3:5) + quadratic;
    end
    cubic = [4 3 2 1].*quartic(1:4);
    if isinf(cubic(1))
        alpha = 0;
        if showstatus && shownow
            fprintf(' decrement=0');
        end
    else
        alphas = roots(cubic); 
        alphas = [alphas(imag(alphas)==0); 0.0]; % selecting real roots only, so imaginary roots don't actually get a lower value
        [fval_next, index] = min(polyval(quartic,alphas));
        decrement = fvals(i) - fval_next;
        if showstatus && shownow
            fprintf(' decrement=%g', decrement);
        end
        alpha = alphas(index);
    end
    Xnew = X+alpha*S;
    X = Xnew;
    clear Xnew;
    if doJerr && ~themodes
        J = X*X';
    end
    if showstatus && shownow
        fprintf('\n');
    end
end

end

