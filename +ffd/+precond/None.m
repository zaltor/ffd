classdef None < handle
    %ffd.precond.NONE no preconditioning
    %
    %   This is a dummy preconditioner that does nothing to the gradient.

    methods
        function [Ghat, reset_cg, Ghat_previous] = subsref(~, args)
            env = args.subs{1};
            Ghat = env.state.G;
            reset_cg = false;
            Ghat_previous = env.state.Ghat_previous;
        end
    end
    
end

