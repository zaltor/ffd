classdef StructHandle < handle
    %STRUCTHANDLE - A struct that is passed by reference
    
    properties
        struct_;
    end
    
    methods
        function obj = StructHandle(varargin)
            obj.struct_ = struct(varargin{:});
        end
        
        function B=subsref(A,S)
            if isequal(substruct('.','struct_'),S)
                B=A.struct_;
            else
                B=subsref(A.struct_,S);
            end
        end
        
        function A=subsasgn(A,S,B)
            if isequal(substruct('.','struct_'),S)
                A.struct_ = B;
            else
                A.struct_=subsasgn(A.struct_,S,B);
            end
        end
        
        function disp(X)
            disp(X.struct_);
        end
    end
    
end

