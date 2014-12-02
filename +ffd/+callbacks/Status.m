classdef Status < handle
    %ffd.callbacks.STATUS Simple status callbacks
    %
    %   ffd.callbacks.STATUS(SEC) creates a callback that displays
    %   simple per-iteration status when at least SEC seconds has
    %   transpired since the last status display.  If SEC is missing, then
    %   a default value of 1 second is assumed.
    
    properties
        last_time = 0;
        sec = 1;
    end
    
    methods
        function obj = Status(sec)
            if exist('sec','var')
                obj.sec = sec;
            end
        end
        
        function subsref(obj, args)
            opts = args.subs{1};
            % consts = args.subs{2}; % not used
            state = args.subs{3};
            if ~opts.verbose
                return;
            end
            if state.t > obj.last_time + obj.sec
                if isfield(state,'Jerr')
                    fprintf('%4d: fval=%10.4e(%10.4e) yerr=%10.4e Jerr=%10.4e Gmag=%10.4e Smag=%10.4e dec=%10.4e\n',...
                        state.iteration, ...
                        state.fval, state.fval_pre, ...
                        state.yerr, state.Jerr, ...
                        sqrt(state.G(:)'*state.G(:)), ...
                        sqrt(state.S(:)'*state.S(:)), ...
                        state.decrement);
                else
                    fprintf('%4d: fval=%10.4e(%10.4e) yerr=%10.4e Gmag=%10.4e Smag=%10.4e dec=%10.4e\n',...
                        state.iteration, ...
                        state.fval, state.fval_pre, ...
                        state.yerr, ...
                        sqrt(state.G(:)'*state.G(:)), ...
                        sqrt(state.S(:)'*state.S(:)), ...
                        state.decrement);
                end
                obj.last_time = state.t;
            end
        end
    end
    
end

