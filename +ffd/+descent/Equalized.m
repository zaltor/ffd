classdef Equalized < handle
    % ffd.descent.Equalized Equalized mode energy descent
    
    properties
        allocated=false;
        KHU_block=[];
        KHUS_block=[];
        y_block=[];
        w2_block=[];
        delta_block=[];
        yerr_mask_block=[];
        Qxx_block=[];
        DU=[];
    end
    
    methods
        
        function allocate(obj, env)
            % allocate temporary storage
            maxBlockSizeYp = max(env.consts.ypIdx2-env.consts.ypIdx1+1);
            maxBlockSizeY = max(env.consts.yIdx2-env.consts.yIdx1+1);
            obj.KHU_block = zeros(maxBlockSizeYp,env.consts.Xsize(2));
            obj.KHUS_block = zeros(maxBlockSizeYp,env.consts.Xsize(2));
            obj.y_block = zeros(maxBlockSizeY,1);
            obj.w2_block = zeros(maxBlockSizeY,1);
            obj.delta_block = zeros(maxBlockSizeY,1);
            obj.yerr_mask_block = false(maxBlockSizeY,1);
            maxBlockSizeQxx = max(env.consts.QxxIdx2-env.consts.QxxIdx1+1);
            obj.Qxx_block = zeros(maxBlockSizeQxx,env.consts.Xsize(2));
            obj.DU = zeros(env.consts.Xsize);
            obj.allocated=true;
        end
        
        function subsref(obj, args)
            env = args.subs{1};
            
            if ~obj.allocated
                obj.allocate(env);
            end
            
            % perform SVD
            [U,S,V] = svd(env.state.X,'econ');
            diagS = diag(S);
            S = diag(sparse(diagS));
            scale = norm(diagS)/sqrt(numel(diagS));
            
            % ready for accumulation
            obj.DU(:) = 0;
            env.state.fval = 0;
            env.state.fval_pre = 0;
            env.state.yerr = 0;
            
            % compute error in intensity and steepest descent direction
            for yIdx=1:env.consts.yBlocks
                yStart = env.consts.yIdx1(yIdx);
                yEnd = env.consts.yIdx2(yIdx);
                yLen = yEnd-yStart+1;
                ypLen = env.consts.ypIdx2(yIdx)-env.consts.ypIdx1(yIdx)+1;
                obj.KHU_block(1:ypLen,:) = 0;
                for xIdx = 1:env.consts.xBlocks
                    obj.KHU_block(1:ypLen,:) = ...
                        obj.KHU_block(1:ypLen,:) + ...
                        env.consts.KH.forward(yIdx,xIdx,U(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
                end
                obj.KHUS_block = obj.KHU_block * S;
                obj.y_block(1:yLen) = ...
                    env.opts.C.forward(yIdx,yIdx,sum(obj.KHUS_block(1:ypLen,:).*conj(obj.KHUS_block(1:ypLen,:)),2));
                obj.delta_block(1:yLen) = env.consts.y(yStart:yEnd) - obj.y_block(1:yLen);
                obj.w2_block(1:yLen) = env.consts.w2(yStart:yEnd);
                E = diag(sparse(env.opts.C.adjoint(yIdx,yIdx,obj.delta_block(1:yLen).*obj.w2_block(1:yLen))));
                for xIdx = 1:env.consts.xBlocks
                    obj.DU(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:) = ...
                        obj.DU(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:) + ...
                            2*env.consts.KH.adjoint(yIdx,xIdx,E*obj.KHU_block(1:ypLen,:));
                end
                % delta_block is magnitude squared from now on
                obj.delta_block(1:yLen) = obj.delta_block(1:yLen).*conj(obj.delta_block(1:yLen));
                obj.yerr_mask_block(1:yLen) = env.opts.yerr_mask(yStart:yEnd);
                env.state.fval = env.state.fval + obj.w2_block(1:yLen)'*obj.delta_block(1:yLen);
                env.state.fval_pre = env.state.fval_pre + obj.w2_block(1:yLen)'*(obj.delta_block(1:yLen).*obj.yerr_mask_block(1:yLen));
                env.state.yerr = env.state.yerr + obj.delta_block(1:yLen)'*obj.yerr_mask_block(1:yLen);
            end

            % compute the rms error
            env.state.yerr = sqrt(env.state.yerr/sum(env.opts.yerr_mask));
            
            % add quadratic (regularizer) component to merit function
            for QxxIdx=1:env.consts.QxxBlocks
                QxxLen = env.consts.QxxIdx2(QxxIdx)-env.consts.QxxIdx1(QxxIdx)+1;                
                obj.Qxx_block(1:QxxLen,:) = 0;
                for xxIdx=1:env.consts.xxBlocks
                    obj.Qxx_block(1:QxxLen,:) = obj.Qxx_block(1:QxxLen,:) + ...
                        env.opts.Q.forward(QxxIdx,xxIdx,U(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
                end
                for xxIdx=1:env.consts.xxBlocks
                    obj.DU(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:) = ...
                        obj.DU(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:) - ...
                            env.opts.Q.adjoint(QxxIdx,xxIdx,obj.Qxx_block(1:QxxLen,:));
                end
                % apply S to calculate actual merit function value
                obj.Qxx_block(1:QxxLen,:) = obj.Qxx_block(1:QxxLen,:)*S;
                env.state.fval = env.state.fval + sum(sum(obj.Qxx_block(1:QxxLen,:).*obj.Qxx_block(1:QxxLen,:)));
            end

            % calculate descent directions
            env.state.G(:,:) = obj.DU * S * V';
            env.state.Ghat(:) = obj.DU * scale * V';

            % always reset cg on first iteration
            if env.state.iteration == 1
                env.state.reset_cg = true;
            end
            
        end
        
    end
end