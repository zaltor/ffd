classdef Blockwise < handle
    %BLOCKWISE Abstract superclass for blockwise linear operators
    %  This class defines an interface for specifying (blockwise) linear
    %  operators. The mental model used is a blockwise matrix, although
    %  implementation can be done using any linear function, as long as
    %  forward and adjoint operations for each "submatrix" is defined.
    %  Derive a new class from this superclass and implement the following
    %  abstract properties and methods to specify your own operator.
    %
    %Abstract 
    %BLOCKWISE properties:
    %  m         - number of rows (i.e. number of output values)
    %  n         - number of columns (i.e. number of input values)
    %  rowSplits - when rows are partitioned, index of the first row in 
    %              each block after the first block (column vector)
    %  colSplits - when columns are partitioned, index of the first column
    %              in each block after the first block (row vector)
    %
    %Abstract
    %BLOCKWISE methods:
    %  forward(obj,s,t,xBlock) - apply submatrix for row block s and column
    %                            block t to xBlock
    %  adjoint(obj,s,t,yBlock) - apply conjugate transpose of submatrix for
    %                            row block s and column block t to yBlock
    %
    %Helper
    %BLOCKWISE methods:
    %  rowBlocks - Return the number of partitions along rows
    %  rowFirst  - Return the first row of block(s)
    %  rowLast   - Return the last row of block(s)
    %  colBlocks - Return the number of partitions along columns
    %  colFirst  - Return the first column of block(s)
    %  colLast   - Return the last column of block(s)

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
    
    methods (Hidden)
        % Hide these unused handle-related functions
        function el = addlistener(varargin)
            el = addlistener@handle(varargin{:});
        end
        function delete(varargin)
            delete@handle(varargin{:});
        end
        function hm = findobj(varargin)
            hm = findobj@handle(varargin{:});
        end
        function p = findprop(varargin)
            p = findprop@handle(varargin{:});
        end
        function notify(varargin)
            notify@handle(varargin{:});
        end
    end
        
    methods (Hidden)
        % Remove handle comparisons
        function eq(varargin)
            error('linops:Blockwise:NotComparable',...
                  'block-wise linear operators are not comparable');
        end
        function ge(varargin)
            error('linops:Blockwise:NotComparable',...
                  'block-wise linear operators are not comparable');
        end
        function gt(varargin)
            error('linops:Blockwise:NotComparable',...
                  'block-wise linear operators are not comparable');
        end
        function le(varargin)
            error('linops:Blockwise:NotComparable',...
                  'block-wise linear operators are not comparable');
        end
        function lt(varargin)
            error('linops:Blockwise:NotComparable',...
                  'block-wise linear operators are not comparable');
        end
        function ne(varargin)
            error('linops:Blockwise:NotComparable',...
                  'block-wise linear operators are not comparable');
        end
    end
    
    methods
        function dims = size(obj, dim)
            %SIZE   Number of inputs and outputs in the linear operator
            %   A.SIZE() or SIZE(A) will return the size of the blockwise
            %   linear operator object A as [M N], where M is the number 
            %   of rows (outputs) and N is the number of columns (inputs).
            if ~exist('dim','var')
                dims = [obj.m, obj.n];
            else
                switch(dim)
                    case 1
                        dims = obj.m;
                    case 2
                        dims = obj.n;
                    otherwise
                        if dim > 2
                            dims = 1;
                        else
                            error('linops:Blockwise:size:InvalidDim',...
                                  'invalid dimension');
                        end
                end
            end
        end
        
        function A = vertcat(varargin)
            %VERTCAT Multiplexing outputs of operators
            %   VERTCAT(A1,A2,...) will combine blockwise linear operators
            %   A1, A2, ... into a single operator by concatenating their 
            %   outputs (rows) as long as all of A1, A2, ... have the same 
            %   column block pattern.
            A = linops.MultiplexedOp(varargin');
        end
        
        function A = horzcat(varargin)
            %HORZCAT Multiplexing inputs of operators
            %   HORZCAT(A1,A2,...) will combine blockwise linear operators
            %   A1, A2, ... into a single operator by concatenating their 
            %   inputs (columns) as long as all of A1, A2, ... have the 
            %   same row block pattern.
            A = linops.MultiplexedOp(varargin);
        end
        
        function A = cat(dim, varargin)
            %CAT     Multiplexing operators through concatenation semantics
            %   CAT(1,A1,A2,...) concatenates the outputs (rows) of the
            %   operators A1, A2, ...
            %
            %   CAT(2,A1,A2,...) concatenates the inputs (columns) of the
            %   operators A1, A2, ...
            %
            %   Concatenation along other dimensions is not supported.
            %   Please see the documentation for VERTCAT and HORZCAT for
            %   more information.
            switch(dim)
                case 1
                    A = linops.MultiplexedOp(varargin');
                case 2
                    A = linops.MultiplexedOp(varargin);
                otherwise
                    error('linops:Blockwise:cat:DimNotSupported',...
                          'concatenation not supported along dimension %d',...
                          dim);
            end
        end
        
        function blocks = rowBlocks(obj)
            %ROWBLOCKS Return the number of partitions along rows
            %   A.ROWBLOCKS returns the number of partitions along rows in 
            %   blockwise linear operator A 
            blocks = numel(obj.rowSplits)+1;
        end
        
        function row = rowFirst(obj, idx)
            %ROWFIRST Return the first row of block(s)
            %   A.ROWFIRST(IDX) fetches the index of the first row(s) of A
            %   in blocks specified by row block indexes IDX
            temp = [1; obj.rowSplits];
            row = temp(idx);
        end
        
        function row = rowLast(obj, idx)
            %ROWLAST Return the last row of block(s)
            %   A.ROWLAST(IDX) fetches the index of the last row(s) of A
            %   in blocks specified by row block indexes IDX
            temp = [obj.rowSplits-1; obj.m];
            row = temp(idx);
        end
        
        function blocks = colBlocks(obj)
            %ROWBLOCKS Return the number of partitions along columns
            %   A.ROWBLOCKS returns the number of partitions along columns
            %   in blockwise linear operator A 
            blocks = numel(obj.colSplits)+1;
        end
        
        function col = colFirst(obj, idx)
            %COLFIRST Return the first column of block(s)
            %   A.COLFIRST(IDX) fetches the index of the first column(s) of
            %   A in blocks specified by row block indexes IDX
            temp = [1, obj.colSplits];
            col = temp(idx);
        end
        
        function col = colLast(obj, idx)
            %COLLAST Return the last column of block(s)
            %   A.COLLAST(IDX) fetches the index of the last column(s) of
            %   A in blocks specified by row block indexes IDX
            temp = [obj.colSplits-1, obj.n];
            col = temp(idx);
        end
    end
end

