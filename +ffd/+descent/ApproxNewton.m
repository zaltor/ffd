classdef ApproxNewton < handle
    %ffd.descent.APPROXNEWTON - periodic approximate Newton preconditioner
    %
    %   Periodically (when conjugacy is poor), computes an approximation to 
    %   the diagonal of the Hessian with respect to a modes decomposition.

    properties
        knm_mag2 = [];
        stored_iteration = 0;
        stored_V = [];
        stored_scale = [];
        rhomax = 0.1;
        force_cg = false;
        P = 1;
    end
    
    methods
        function obj = ApproxNewton(rhomax, P, force_cg)
            if exist('rhomax','var')
                obj.rhomax = rhomax;
            end
            if exist('force_cg','var')
                obj.force_cg = force_cg;
            end
            if exist('P','var')
                obj.P = P;
            end
        end
        
        function subsref(obj, args)
            env = args.subs{1};
            
            if isempty(obj.knm_mag2)
                if isequal(obj.P,'ones')
                    obj.knm_mag2 = true(size(env.consts.KH,2),size(env.consts.KH,1));
                    obj.P = 1;
                else
                    fprintf('Computing |k_{mn}|^2\n');
                    obj.knm_mag2(size(env.consts.KH,2),size(env.consts.KH,1)) = 0;
                    for yIdx=1:env.consts.yBlocks
                        yStart = env.consts.yIdx1(yIdx);
                        yEnd = env.consts.yIdx2(yIdx);
                        for nOff=0:env.opts.R:(env.consts.N-1)
                            nStart = 1+nOff;
                            nEnd = min(env.consts.N,env.opts.R+nOff);
                            U = zeros(env.consts.N,nEnd-nStart+1);
                            U(nStart:nEnd,:)=eye(nEnd-nStart+1);
                            U = obj.P*U;
                            KHU_block = env.consts.KH.forward(yIdx,1,U(env.consts.xIdx1(1):env.consts.xIdx2(1),:));
                            for xIdx = 2:env.consts.xBlocks
                                KHU_block = ...
                                    KHU_block + env.consts.KH.forward(yIdx,xIdx,U(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
                            end
                            obj.knm_mag2(nStart:nEnd,yStart:yEnd) = (real(KHU_block).^2 + imag(KHU_block).^2)';
                        end
                    end
                end
            end
            
            % allocate memory
            DU = zeros(env.consts.Xsize);
            DUblock(env.consts.Xsize(1),env.consts.Xsize(2)) = 0;
            scale = zeros(env.consts.N,env.opts.R);
            
            % compute SVD
            [U,S,V] = svd(env.state.X,'econ');
            diagS = diag(S);
            
            % ready for accumulation
            env.state.fval = 0;
            env.state.fval_pre = 0;
            env.state.yerr = 0;

            % compute error in intensity and steepest descent direction
            for yIdx=1:env.consts.yBlocks
                yStart = env.consts.yIdx1(yIdx);
                yEnd = env.consts.yIdx2(yIdx);
                KHU_block = env.consts.KH.forward(yIdx,1,U(env.consts.xIdx1(1):env.consts.xIdx2(1),:));
                for xIdx = 2:env.consts.xBlocks
                    KHU_block = ...
                        KHU_block + env.consts.KH.forward(yIdx,xIdx,U(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
                end
                KHUS_block = bsxfun(@times,real(KHU_block).^2+imag(KHU_block).^2,(diagS.^2)');
                y_block = ...
                    env.opts.C.forward(yIdx,yIdx,sum(KHUS_block,2));
                delta_block = env.consts.y(yStart:yEnd) - y_block;
                w2_block = env.consts.w2(yStart:yEnd);
                scale = scale + obj.knm_mag2(:,yStart:yEnd) * bsxfun(@times,bsxfun(@minus,KHUS_block,delta_block),env.opts.C.adjoint(yIdx,yIdx,w2_block));
                % KHU_block getting multiplied on the left by E
                KHU_block = bsxfun(@times,env.opts.C.adjoint(yIdx,yIdx,delta_block.*w2_block),KHU_block);
                % delta_block is magnitude squared from now on
                delta_block = delta_block.*conj(delta_block);
                yerr_mask_block = env.opts.masky(yStart:yEnd);
                env.state.fval = env.state.fval + w2_block'*delta_block;
                env.state.fval_pre = env.state.fval_pre + w2_block'*(delta_block.*yerr_mask_block);
                env.state.yerr = env.state.yerr + delta_block'*yerr_mask_block;
                for xIdx = 1:env.consts.xBlocks
                    xStart = env.consts.xIdx1(xIdx);
                    xEnd = env.consts.xIdx2(xIdx);
                    DUblock(xStart:xEnd,:) = env.consts.KH.adjoint(yIdx,xIdx,KHU_block);
                end
                DU = DU + 2*DUblock;
            end

            % compute the rms error
            env.state.yerr = sqrt(env.state.yerr/sum(env.opts.masky));
            
            % add quadratic (regularizer) component to merit function
            % TODO: skip for now
            for QxxIdx=1:env.consts.QxxBlocks
                Qxx_block = env.opts.Q.forward(QxxIdx,1,U(env.consts.xxIdx1(1):env.consts.xxIdx2(1),:));
                for xxIdx=2:env.consts.xxBlocks
                    Qxx_block = Qxx_block + ...
                        env.opts.Q.forward(QxxIdx,xxIdx,U(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
                end
                env.state.fval = env.state.fval + Qxx_block(:)'*Qxx_block(:);
                C2 = C2 + sum(Qxx_block.*conj(Qxx_block),1)';
                for xxIdx=1:env.consts.xxBlocks
                    xxStart = env.consts.xxIdx1(xxIdx);
                    xxEnd = env.consts.xxIdx2(xxIdx);
                    DUblock(xxStart:xxEnd,:) = env.opts.Q.adjoint(QxxIdx,xxIdx,Qxx_block);
                end
                DU = DU - DUblock;
            end
            
            % copy out the steepest descent direction
            env.state.G(:) = bsxfun(@times,DU,diagS') * V';
            
            % determine whether to terminate iterations
            tmp = norm(env.state.G,'fro');
            if env.consts.N == env.opts.R && env.opts.B*tmp < env.consts.tau_prime
                UHDU = U'*DU;
                eta = eigs(UHDU,1,'LR');
                env.state.terminate = env.opts.B*tmp + env.opts.B^2*eta < env.consts.tau_prime;
                if env.state.terminate
                    return;
                else
                    fprintf('.');
                end
            end

            % do rho test to see if we should compute new preconditioner
            env.state.reset_cg = false;
            if obj.stored_iteration == 0
                env.state.reset_cg = true;
            else
                Ghat = obj.P * (obj.stored_scale .* (obj.P'*(env.state.G*obj.stored_V)))*obj.stored_V';
                cur_rho = abs( real(env.state.G_previous(:)'*Ghat(:)) / real(env.state.G(:)'*Ghat(:)) );
                if cur_rho > obj.rhomax
                    env.state.reset_cg = true;
                end
            end

            if env.state.reset_cg
                % compute distortion correction
                scale = abs(2*scale);
                scale = scale/max(scale(:));
                scale = max(scale, eps*numel(scale));
                scale = 1./scale;
                
                % update state
                obj.stored_scale = scale;
                obj.stored_V = V;
                obj.stored_iteration = env.state.iteration;

                % compute step direction
                env.state.Ghat(:) = obj.P * (obj.stored_scale .* (obj.P'*bsxfun(@times,DU,diagS'))) * obj.stored_V';
            else
                env.state.Ghat(:) = Ghat;
            end
            
            if env.state.iteration ~= 1
                env.state.reset_cg = (~obj.force_cg) && env.state.reset_cg;
            end
        end
    end
end

