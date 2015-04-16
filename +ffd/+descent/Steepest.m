classdef Steepest
    % ffd.descent.STEEPEST steepest descent
    
    methods
        
        function subsref(~, args)
            env = args.subs{1};
            
            % allocate memory
            G = zeros(env.consts.Xsize);
            Gblock(env.consts.Xsize(1),env.consts.Xsize(2)) = 0;
            
            % ready for accumulation
            env.state.fval = 0;
            env.state.fval_pre = 0;
            env.state.yerr = 0;

            % compute error in intensity and steepest descent direction
            for yIdx=1:env.consts.yBlocks
                yStart = env.consts.yIdx1(yIdx);
                yEnd = env.consts.yIdx2(yIdx);
                KHX_block = env.consts.KH.forward(yIdx,1,env.state.X(env.consts.xIdx1(1):env.consts.xIdx2(1),:));
                for xIdx = 2:env.consts.xBlocks
                    KHX_block = ...
                        KHX_block + env.consts.KH.forward(yIdx,xIdx,env.state.X(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
                end
                y_block = ...
                    env.opts.C.forward(yIdx,yIdx,sum(real(KHX_block).^2,2)+sum(imag(KHX_block).^2,2));
                delta_block = env.consts.y(yStart:yEnd) - y_block;
                w2_block = env.consts.w2(yStart:yEnd);
                % KHX_block getting multiplied on the left by E
                KHX_block = bsxfun(@times,env.opts.C.adjoint(yIdx,yIdx,delta_block.*w2_block),KHX_block);
                % delta_block is magnitude squared from now on
                delta_block = delta_block.*conj(delta_block);
                yerr_mask_block = env.opts.masky(yStart:yEnd);
                env.state.fval = env.state.fval + w2_block'*delta_block;
                env.state.fval_pre = env.state.fval_pre + w2_block'*(delta_block.*yerr_mask_block);
                env.state.yerr = env.state.yerr + delta_block'*yerr_mask_block;
                for xIdx = 1:env.consts.xBlocks
                    xStart = env.consts.xIdx1(xIdx);
                    xEnd = env.consts.xIdx2(xIdx);
                    Gblock(xStart:xEnd,:) = env.consts.KH.adjoint(yIdx,xIdx,KHX_block);
                end
                G = G + 2*Gblock;
            end

            % compute the rms error
            env.state.yerr = sqrt(env.state.yerr/sum(env.opts.masky));
            
            % add quadratic (regularizer) component to merit function
            for QxxIdx=1:env.consts.QxxBlocks
                Qxx_block = env.opts.Q.forward(QxxIdx,1,env.state.X(env.consts.xxIdx1(1):env.consts.xxIdx2(1),:));
                for xxIdx=2:env.consts.xxBlocks
                    Qxx_block = Qxx_block + ...
                        env.opts.Q.forward(QxxIdx,xxIdx,env.state.X(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
                end
                env.state.fval = env.state.fval + Qxx_block(:)'*Qxx_block(:);
                for xxIdx=1:env.consts.xxBlocks
                    xxStart = env.consts.xxIdx1(xxIdx);
                    xxEnd = env.consts.xxIdx2(xxIdx);
                    Gblock(xxStart:xxEnd,:) = env.opts.Q.adjoint(QxxIdx,xxIdx,Qxx_block);
                end
                G = G - Gblock;
            end

            % copy out the descent direction
            env.state.G(:) = G;
            env.state.Ghat(:) = G;

            % always reset cg on first iteration
            if env.state.iteration == 1
                env.state.reset_cg = true;
            end
            
        end
        
    end
end
