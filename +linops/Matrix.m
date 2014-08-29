classdef Matrix < linops.Blockwise
    %MATRIX Matrix-based blockwise linear operator
    %   Detailed explanation goes here
    
    properties
        m;
        n;
        rowSplits;
        colSplits;
    end
    
    properties (Hidden)
        A;
        rowStarts;
        rowEnds;
        colStarts;
        colEnds;
    end
    
    methods
        function obj = Matrix(A, maxRows, maxCols)
            dims = size(A);
            obj.m = dims(1);
            obj.n = dims(2);
            obj.A = A;
            % by default allow up to 1 million elements in submatrix
            % and only split by rows
            if ~exist('maxRows','var')
                maxRows = 1e6 / obj.n;
            end
            if ~exist('maxCols','var')
                maxCols = Inf;
            end
            nRowBlocks = ceil(obj.m/maxRows);
            nColBlocks = ceil(obj.n/maxCols);
            obj.rowEnds = round((1:nRowBlocks)*obj.m/nRowBlocks);
            obj.colEnds = round((1:nColBlocks)*obj.n/nColBlocks);
            obj.rowSplits = obj.rowEnds(1:end-1)+1;
            obj.colSplits = obj.colEnds(1:end-1)+1;
            obj.rowStarts = [1,obj.rowSplits];
            obj.colStarts = [1,obj.colSplits];
        end
        
        function yBlock = forward(obj, s, t, xBlock)
            yBlock = obj.A(obj.rowStarts(s):obj.rowEnds(s),...
                           obj.colStarts(t):obj.colEnds(t)) * xBlock;
        end
        
        function xBlock = adjoint(obj, s, t, yBlock)
            xBlock = obj.A(obj.rowStarts(s):obj.rowEnds(s),...
                           obj.colStarts(t):obj.colEnds(t))' * yBlock;
        end
    end
    
end

