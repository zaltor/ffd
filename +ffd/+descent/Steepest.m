classdef Steepest < handle
    % ffd.descent.Steepest Steepest descent
    
    methods
        
        function subsref(~, args)
            env = args.subs{1};
            
            % ready G for accumulation
            env.state.G(:) = 0;
            
            % prepare delta if needed
            if numel(env.state.delta) ~= env.consts.M
                env.state.delta = zeros(env.consts.M,1);
            end

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

            % just steepest descent
            %env.state.Ghat(:) = env.state.G(:);
            
            [env.state.Ghat, env.state.reset_cg, env.state.Ghat_previous] = env.opts.precond(env);

            % always reset cg on first iteration
            if env.state.iteration == 1
                env.state.reset_cg = true;
            end
            
        end
        
    end
end