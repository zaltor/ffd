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
            env = args.subs{1};
            if ~env.opts.verbose
                return;
            end
            if env.state.t > obj.last_time + obj.sec
                if isfield(env.state,'Jerr')
                    fprintf('%4d: fval=%10.4e(%10.4e) yerr=%10.4e Jerr=%10.4e Gmag=%10.4e Smag=%10.4e dec=%10.4e\n',...
                        env.state.iteration, ...
                        env.state.fval, env.state.fval_pre, ...
                        env.state.yerr, env.state.Jerr, ...
                        sqrt(env.state.G(:)'*env.state.G(:)), ...
                        sqrt(env.state.S(:)'*env.state.S(:)), ...
                        env.state.decrement);
                else
                    fprintf('%4d: fval=%10.4e(%10.4e) yerr=%10.4e Gmag=%10.4e Smag=%10.4e dec=%10.4e\n',...
                        env.state.iteration, ...
                        env.state.fval, env.state.fval_pre, ...
                        env.state.yerr, ...
                        sqrt(env.state.G(:)'*env.state.G(:)), ...
                        sqrt(env.state.S(:)'*env.state.S(:)), ...
                        env.state.decrement);
                end
                obj.last_time = env.state.t;
            end
        end
    end
    
end

