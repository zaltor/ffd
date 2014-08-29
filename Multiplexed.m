classdef Multiplexed < linops.Blockwise
    %MULTIPLEXED A blockwise linear operator containing sub-operators
    
    properties
        m;
        n;
        rowSplits;
        colSplits;
    end
    
    properties (Hidden)
        sources; % cell matrix of component operators
        lookup; % lookup table to map blocks to sources
    end
    
    methods
        function obj = Multiplexed(sources)
            obj.sources = sources;
            obj.update;
        end
        
        function update(obj)
            %UPDATE Make sure properties are up to date
            srcDims = size(obj.sources);
            % populate m and rowSplits and check for consistency
            for j=1:srcDims(2)
                mTemp = 0;
                rowEndsTemp = [];
                for i=1:srcDims(1)
                    src = obj.sources{i,j};
                    if ismethod(src,'update') src.update; end
                    rowEndsTemp = [rowEndsTemp; 
                                   mTemp+src.rowSplits-1; 
                                   mTemp+src.m];
                    mTemp = mTemp + src.m;
                end
                if j==1
                    % populate
                    obj.m = mTemp;
                    obj.rowSplits = rowEndsTemp(1:end-1)+1;
                else
                    % check for consistency
                    if obj.m ~= mTemp
                        error('linops:Multiplexed:update:InconsistentRows',...
                              'inconsistent row count for column %d in sources',...
                              j);
                    end
                    if ~isequal(obj.rowSplits,rowEndsTemp(1:end-1)+1)
                        error('linops:Multiplexed:update:InconsistentRowSplits',...
                              'inconsistent row blocks for column %d in sources',...
                              j);
                    end  
                end
            end
            % populate n and colSplits and check for consistency
            for i=1:srcDims(1)
                nTemp = 0;
                colEndsTemp = [];
                for j=1:srcDims(2)
                    src = obj.sources{i,j};
                    colEndsTemp = [colEndsTemp,...
                                   nTemp+src.colSplits-1,...
                                   nTemp+src.n];
                    nTemp = nTemp + src.n;
                end
                if i==1
                    % populate
                    obj.n = nTemp;
                    obj.colSplits = colEndsTemp(1:end-1)+1;
                else
                    % check for consistency
                    if obj.n ~= nTemp
                        error('linops:Multiplexed:update:InconsistentCols',...
                              'inconsistent column count for row %d in sources',...
                              i);
                    end
                    if ~isequal(obj.colSplits,colEndsTemp(1:end-1)+1)
                        error('linops:Multiplexed:update:InconsistentColSplits',...
                              'inconsistent column blocks for row %d in sources',...
                              i);
                    end  
                end
            end
            % populate lookup table for each block
            obj.lookup = cell(numel(obj.rowSplits),numel(obj.colSplits));
            rowBlockOffsets = zeros(1,numel(obj.colSplits)+1);
            for i=1:srcDims(1)
                colBlockOffset = 0;
                for j=1:srcDims(2)
                    src = obj.sources{i,j};
                    nRowBlocks = length(src.rowSplits)+1;
                    nColBlocks = length(src.colSplits)+1;
                    for ii=1:nRowBlocks
                        for jj=1:nColBlocks
                            rowBlockOffsets(colBlockOffset+jj) = ...
                                rowBlockOffsets(colBlockOffset+jj) + 1;
                            obj.lookup{rowBlockOffsets(colBlockOffset+jj),...
                                       colBlockOffset+jj} = [i,j,ii,jj];
                        end
                    end
                    colBlockOffset = colBlockOffset + nColBlocks;
                end
            end
            % done
        end
        
        function yBlock = forward(obj, s, t, xBlock)
            indexes = obj.lookup(s,t);
            src = obj.sources(indexes(1),indexes(2));
            yBlock = src.forward(indexes(3),indexes(4),xBlock);
        end
        
        function xBlock = adjoint(obj, s, t, yBlock)
            indexes = obj.lookup(s,t);
            src = obj.sources(indexes(1),indexes(2));
            xBlock = src.adjoint(indexes(3),indexes(4),yBlock);
        end

    end
    
end

