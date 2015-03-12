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
            rowEnds = round((1:nRowBlocks)*obj.m/nRowBlocks);
            colEnds = round((1:nColBlocks)*obj.n/nColBlocks);
            obj.rowSplits = rowEnds(1:end-1)+1;
            obj.colSplits = colEnds(1:end-1)+1;
            obj.rowSplits = obj.rowSplits(:);
            obj.colSplits = obj.colSplits(:)';
            
            obj.update;
        end
        
        function yBlock = forward(obj, s, t, xBlock)
            yBlock = obj.A(obj.rowFirst(s):obj.rowLast(s),...
                           obj.colFirst(t):obj.colLast(t)) * xBlock;
        end
        
        function xBlock = adjoint(obj, s, t, yBlock)
            xBlock = obj.A(obj.rowFirst(s):obj.rowLast(s),...
                           obj.colFirst(t):obj.colLast(t))' * yBlock;
        end
    end
    
end

