classdef Equalize < handle
    %ffd.precond.EQUALIZE preconditioning by equalizing mode energies
    %
    %   Instead of multiplying the mutual intensity space gradient by
    %   the current modes, ffd.precond.EQUALIZE multiples the mutual
    %   intensity space gradient by a set of modes with the current mode
    %   energies redistributed evenly.

    methods
        function [Ghat, reset_cg, Ghat_previous] = subsref(~, args)
            env = args.subs{1};

            % setup
            Ghat = zeros(env.consts.Xsize);

            % first, equalize mode energies
            [U_,S_,V_] = svd(env.state.X, 'econ');
            if S_(1,1) == 0
                % X is the zero matrix, so make up something
                if env.consts.Xsize(1) > env.consts.Xsize(2)
                    X_fake = repmat(eye(env.consts.Xsize(2)),[ceil(env.consts.Xsize(1)/env.consts.Xsize(2)),1]);
                else
                    X_fake = repmat(eye(env.consts.Xsize(1)),[1,ceil(env.consts.Xsize(2)/env.consts.Xsize(1))]);
                end
                precondX = X_fake(1:env.consts.Xsize(1),1:env.consts.Xsize(2));
                clear X_fake;
            else
                new_S = sqrt(trace(S_.^2)/size(S_,1))*eye(size(S_,1));
                precondX = U_*new_S*V_';
                clear new_S;
            end
            clear U_;
            clear S_;
            clear V_;

            % now compute Ghat, first for the quartic portion
            for yIdx=1:env.consts.yBlocks
                KHprecondX_block = 0;
                w2_block = env.consts.w2(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx));
                delta_block = env.state.delta(env.consts.yIdx1(yIdx):env.consts.yIdx2(yIdx));
                E = diag(sparse(env.opts.C.adjoint(yIdx,yIdx,delta_block.*w2_block)));
                for xIdx = 1:env.consts.xBlocks
                    KHprecondX_block = KHprecondX_block + env.consts.KH.forward(yIdx,xIdx,precondX(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:));
                    Ghat(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:) = ...
                        Ghat(env.consts.xIdx1(xIdx):env.consts.xIdx2(xIdx),:) + ...
                            4*env.consts.KH.adjoint(yIdx,xIdx,E*KHprecondX_block);
                end
                clear KHprecondX_block;
            end

            % then update for the quadratic portion
            for QxxIdx=1:env.consts.QxxBlocks
                Qxx_block = 0;
                for xxIdx=1:env.consts.xxBlocks
                    Qxx_block = Qxx_block + ...
                        env.opts.Q.forward(QxxIdx,xxIdx,precondX(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:));
                end
                for xxIdx=1:env.consts.xxBlocks
                    Ghat(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:) = ...
                        Ghat(env.consts.xxIdx1(xxIdx):env.consts.xxIdx2(xxIdx),:) - ...
                            2*env.opts.Q.adjoint(QxxIdx,xxIdx,Qxx_block);
                end
                clear Qxx_block;
            end
            
            reset_cg = false;
            Ghat_previous = env.state.Ghat_previous;
        end
    end
end

