classdef Equalized
    % ffd.descent.EQUALIZED equalized energy descent
    
    methods
        
        function subsref(~, args)
            env = args.subs{1};
            
            % allocate memory
            DU = zeros(env.consts.Xsize);
            DUblock(env.consts.Xsize(1),env.consts.Xsize(2)) = 0;
            
            % compute SVD
            [U,S,V] = svd(env.state.X,'econ');
            diagS = diag(S);
            scale = sqrt(diagS'*diagS/numel(diag(S)));
            
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
                y_block = env.opts.C.forward(yIdx,yIdx,(real(KHU_block).^2)*(diagS.^2)+(imag(KHU_block).^2)*(diagS.^2));
                delta_block = env.consts.y(yStart:yEnd) - y_block;
                w2_block = env.consts.w2(yStart:yEnd);
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
            for QxxIdx=1:env.consts.QxxBlocks
                Qxx_block = env.opts.Q.forward(QxxIdx,1,U(env.consts.xxIdx1(1):env.consts.xxIdx2(1),:));
                for xxIdx=2:env.consts.xxBlocks
                    Qxx_block = Qxx_block + ...
                        env.opts.Q.forward(QxxIdx,xxIdx,U(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
                end
                env.state.fval = env.state.fval + Qxx_block(:)'*Qxx_block(:);
                for xxIdx=1:env.consts.xxBlocks
                    xxStart = env.consts.xxIdx1(xxIdx);
                    xxEnd = env.consts.xxIdx2(xxIdx);
                    DUblock(xxStart:xxEnd,:) = env.opts.Q.adjoint(QxxIdx,xxIdx,Qxx_block);
                end
                DU = DU - DUblock;
            end

            % copy out the descent direction
            env.state.G(:) = bsxfun(@times,DU,diagS') * V';
            
            % determine whether to terminate iterations
            tmp = norm(env.state.G,'fro');
            if env.consts.N == env.opts.R && env.opts.B*tmp < env.consts.tau_prime
                UHDU = U'*DU;
                eta = eigs(UHDU,1,'LR');
                env.state.terminate = env.opts.B*tmp + env.opts.B^2*eta < env.consts.tau_prime;
                if env.state.terminate
                    return;
                end
            end
            
            % copy out step direction
            env.state.Ghat(:) = scale*(DU*V');

            % always reset cg on first iteration
            if env.state.iteration == 1
                env.state.reset_cg = true;
            end
            
        end
        
    end
end
