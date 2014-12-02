classdef Equalize < handle
    %ffd.precond.EQUALIZE preconditioning by equalizing mode energies
    %
    %   Instead of multiplying the mutual intensity space gradient by
    %   the current modes, ffd.precond.EQUALIZE multiples the mutual
    %   intensity space gradient by a set of modes with the current mode
    %   energies redistributed evenly.

    methods
        function Ghat = subsref(~, args)
            opts = args.subs{1};
            consts = args.subs{2};
            state = args.subs{3};

            % setup
            Ghat = zeros(consts.Xsize);

            % first, equalize mode energies
            [U_,S_,V_] = svd(state.X, 'econ');
            if S_(1,1) == 0
                % X is the zero matrix, so make up something
                if consts.Xsize(1) > consts.Xsize(2)
                    X_fake = repmat(eye(consts.Xsize(2)),[ceil(consts.Xsize(1)/consts.Xsize(2)),1]);
                else
                    X_fake = repmat(eye(consts.Xsize(1)),[1,ceil(consts.Xsize(2)/consts.Xsize(1))]);
                end
                precondX = X_fake(1:consts.Xsize(1),1:consts.Xsize(2));
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
            for yIdx=1:consts.yBlocks
                KHprecondX_block = 0;
                w2_block = consts.w2(consts.yIdx1(yIdx):consts.yIdx2(yIdx));
                delta_block = state.delta(consts.yIdx1(yIdx):consts.yIdx2(yIdx));
                E = diag(sparse(opts.C.adjoint(yIdx,yIdx,delta_block.*w2_block)));
                for xIdx = 1:consts.xBlocks
                    KHprecondX_block = KHprecondX_block + consts.KH.forward(yIdx,xIdx,precondX(consts.xIdx1(xIdx):consts.xIdx2(xIdx),:));
                    Ghat(consts.xIdx1(xIdx):consts.xIdx2(xIdx),:) = ...
                        Ghat(consts.xIdx1(xIdx):consts.xIdx2(xIdx),:) + ...
                            4*consts.KH.adjoint(yIdx,xIdx,E*KHprecondX_block);
                end
                clear KHprecondX_block;
            end

            % then update for the quadratic portion
            for QxxIdx=1:consts.QxxBlocks
                Qxx_block = 0;
                for xxIdx=1:consts.xxBlocks
                    Qxx_block = Qxx_block + ...
                        opts.Q.forward(QxxIdx,xxIdx,precondX(consts.xxIdx1(xxIdx):consts.xxIdx2(xxIdx),:));
                end
                for xxIdx=1:consts.xxBlocks
                    Ghat(consts.xxIdx1(xxIdx):consts.xxIdx2(xxIdx),:) = ...
                        Ghat(consts.xxIdx1(xxIdx):consts.xxIdx2(xxIdx),:) - ...
                            2*opts.Q.adjoint(QxxIdx,xxIdx,Qxx_block);
                end
                clear Qxx_block;
            end
        end
    end
end

