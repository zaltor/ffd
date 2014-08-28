classdef BlockLinOp < handle
    %BLOCKLINOP Abstract superclass for blockwise linear operators
    %  This class defines an interface for specifying (blockwise) linear
    %  operators. The mental model used is a blockwise matrix, although
    %  implementation can be done using any linear function, as long as
    %  forward and adjoint operations for each "submatrix" is defined.
    %  Derive a new class from this superclass and implement the following
    %  abstract properties and methods to specify your own operator.
    %
    %BLOCKLINOP properties:
    %  m          - number of rows (i.e. number of output values)
    %  n          - number of columns (i.e. number of input values)
    %  rowSplits - when rows are partitioned, index of the first row in 
    %              each block after the first block (column vector)
    %  colSplits - when columns are partitioned, index of the first column
    %              in each block after the first block (row vector)
    %
    %BLOCKLINOP methods:
    %  forward(obj,s,t,xBlock) - apply submatrix for row block s and column
    %                            block t to xBlock
    %  adjoint(obj,s,t,yBlock) - apply conjugate transpose of submatrix for
    %                            row block s and column block t to yBlock
    
    properties (Abstract)
        m; % number of rows/output values
        n; % number of columns/input values
        rowSplits; % indexes of first row of each row partition after first
        colSplits; % indexes of first col of each col partition after first
    end
    
    methods (Abstract)
        forward(obj,s,t,xBlock); % single block forward transform 
        adjoint(obj,s,t,yBlock); % single block adjoint transform
    end
    
    methods
        function dims = size(obj)
            %SIZE   Number of inputs and outputs in the linear operator
            dims = [obj.m, obj.n];
        end
        
        function A = vertcat(varargin)
            A = MultiplexedOp(varargin');
        end
        
        function A = horzcat(varargin)
            A = MultiplexedOp(varargin);
        end
        
        function A = cat(dim, varargin)
            switch(dim)
                case 1
                    A = MultiplexedOp(varargin');
                case 2
                    A = MultiplexedOp(varargin);
                otherwise
                    error('BlockLinOp:catdimnotsupported',...
                          'concatenation not supported along dimension %d',...
                          dim);
            end
        end
    end
end

