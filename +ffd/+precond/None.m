classdef None < handle
    %ffd.precond.NONE no preconditioning
    %
    %   This is a dummy preconditioner that does nothing to the gradient.

    methods
        function Ghat = subsref(~, args)
            
            % opts = args.subs{1}; % not used
            % consts = args.subs{2}; % not used
            state = args.subs{3};
            Ghat = state.G;
        end
    end
    
end

