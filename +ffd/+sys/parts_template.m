function fun = parts_template(blocks)
%ffd.sys.PARTS_TEMPLATE    Generate a helper function for block splitting
%   FUN = ffd.sys.PARTS_TEMPLATE(BLOCKS) creates a helper function to
%   implement the sys.parts function in the optical system specification
%   for FFD. BLOCKS is a sequence of integers denoting the size of each
%   block; blocks must be contiguous and must be specified sequentially.

fun = @parts;
boundaries = [0;cumsum(blocks(:))];

    function result = parts(part_number)
        if ~exist('part_number','var')
            result = length(boundaries)-1;
        else
            result = [boundaries(part_number)+1, ...
                      boundaries(part_number+1)];
        end
    end

end

