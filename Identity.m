classdef Identity < linops.Blockwise
    %IDENTITY The identity linear operator
    %   Detailed explanation goes here
    
    properties
        m;
        n;
        rowSplits;
        colSplits;
    end
    
    properties (Hidden)
        blockSizes;
    end
    
    methods
        function obj = Identity(n, splits)
            if ~exist('splits','var')
                splits = [];
            end
            obj.m = n;
            obj.n = n;
            obj.rowSplits = splits;
            obj.colSplits = splits;
            obj.blockSizes = diff([1; splits(:); n+1]);
            if sum(abs(obj.blockSizes)) ~= n
                error('linops:Identity:IncorrectSplits',...
                      'incorrect specification of block boundaries');
            end
        end
        
        function yBlock = forward(obj, s, t, xBlock)
            if size(xBlock,1) ~= obj.blockSizes(t)
                error('linops:Identity:Forward:ColumnBlockMismatch',...
                      'incorrect number of rows in xBlock');
            end
            if s==t
                yBlock = xBlock;
            else
                xBlockSize = size(xBlock);
                yBlock = zeros([obj.blockSizes(s), xBlockSize(2:end)]);
            end
        end
        
        function xBlock = adjoint(obj, s, t, yBlock)
            if size(yBlock,1) ~= obj.blockSizes(s)
                error('linops:Identity:Adjoint:RowBlockMismatch',...
                      'incorrect number of rows in yBlock');
            end
            if s==t
                xBlock = yBlock;
            else
                yBlockSize = size(yBlock);
                xBlock = zeros([obj.blockSizes(t), yBlockSize(2:end)]);
            end
        end
    end
    
end

