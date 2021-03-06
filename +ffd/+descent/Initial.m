classdef Initial 
    %ffd.descent.INITIAL computing an initial descent direction from zero
    
    methods

        function subsref(~, args)
            env = args.subs{1};

            % get orthogonalized version of current X
            [U,S] = svd(env.state.X,'econ');
            
            % set X to 0
            env.state.X(:) = 0;
            
            % make sure to reset cg
            env.state.reset_cg = true;
            
            % if input is actually zero, use a randomized basis
            if S(1,1) == 0
                X = env.opts.rs.randn(env.consts.Xsize) + ...
                    1j*env.opts.rs.randn(env.consts.Xsize);
                [U,~] = svd(X, 'econ');
            end
   
            % compute U^HDU
            DU = 0;
            DUblock(env.consts.Xsize(1),env.consts.Xsize(2)) = 0;
            for yIdx=1:env.consts.yBlocks
                yStart = env.consts.yIdx1(yIdx);
                yEnd = env.consts.yIdx2(yIdx);
                KHU_block = env.consts.KH.forward(yIdx,1,U(env.consts.xIdx1(1):env.consts.xIdx2(1),:));
                for xIdx = 2:env.consts.xBlocks
                    KHU_block = ...
                        KHU_block + env.consts.KH.forward(yIdx,xIdx,U(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
                end
                delta_block = env.consts.y(yStart:yEnd);
                w2_block = env.consts.w2(yStart:yEnd);
                % KH_block getting multiplied on the left by E
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
            UHDU = U'*DU;
            UHDU = 0.5*(UHDU+UHDU');
            
            % factor resulting UHDU (what we actually want is 
            % Ghat*Ghat' = UU^H*D*UU^H (i.e. projection of D onto basis)
            [V,D] = eig(UHDU); % U^HDU = VDV^-1
            D = sqrt(max(real(diag(D)),0));
            
            % set the output directions
            env.state.G(:) = 0;
            env.state.Ghat(:) = U*bsxfun(@times,V,D');
        end
    end
end
                
