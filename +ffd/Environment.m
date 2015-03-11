classdef Environment < handle
    %ffd.ENVIRONMENT Hold one shared copy of state, etc. for ffd
    
    properties
        opts;
        consts;
        state;
    end
end
