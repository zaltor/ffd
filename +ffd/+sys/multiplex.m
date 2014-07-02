function sys = multiplex(varargin)
%ffd.sys.MULTIPLEX    combine multiple optical systems together in parallel
%   SYS = SETUP_FFD_MULTIPLEX(SYS1, SYS2, ..., SYSN) will combine several
%   optical system specificatinos SYS1 through SYSN together into a single
%   specification. All system specifications need to have the same number
%   of input samples (i.e. SYS1.N == SYS2.N == ... == SYSN.N). The
%   resulting optical system specification will have output measurements
%   concatenated.

num_sys = length(varargin);

% check to make sure they all have the same input size
N = varargin{1}.N;
for i=2:num_sys
    if N ~= varargin{i}.N
        error('setup_ffd_multiplex:not_same_N','systems to be multiplexed must have same input size (N)');
    end
end

% count how many partitions
total_parts = 0;
num_parts = zeros(num_sys,1);
for i=1:num_sys
    num_parts(i) = varargin{i}.parts();
    total_parts = total_parts + num_parts(i);
end

% create partition map
part_map = zeros(total_parts,4);
offset_part = 0;
offset_y = 0;
for i=1:num_sys
    part_map((1:num_parts(i))+offset_part,1) = i;
    part_map((1:num_parts(i))+offset_part,2) = 1:num_parts(i);
    for j=1:num_parts(i)
        part_map(offset_part+j,3:4) = offset_y + varargin{i}.parts(j);
    end
    offset_y = max(part_map((1:num_parts(i))+offset_part,4));
    offset_part = offset_part + num_parts(i);
end

% create dummy gather/scatter operations if needed
gathers = cell(num_sys,1);
scatters = cell(num_sys,1);
for i=1:num_sys
    try
        M = 0;
        for j=1:varargin{i}.parts()
            M = max(M, max(varargin{i}.parts(j)));
        end
        varargin{i}.gather(varargin{i}.scatter(zeros(M,0),1),1);
        gathers{i} = @(y,ii) varargin{i}.gather(y,ii);
        scatters{i} = @(y,ii) varargin{i}.scatter(y,ii);
    catch err
        gathers{i} = @(y,ii) y;
        scatters{i} = @(y,ii) y;
    end
end

% create new multiplexed system
sys = struct;
sys.parts = @part_helper;
sys.N = N;
sys.forward = @forward_helper;
sys.adjoint = @adjoint_helper;
sys.gather = @gather_helper;
sys.scatter = @scatter_helper;

    function result = part_helper(part_number)
        if ~exist('part_number','var')
            result = total_parts;
        else
            result = part_map(part_number,3:4);
        end
    end

    function Y = forward_helper(X, i)
        sys_id = part_map(i,1);
        part_id = part_map(i,2);
        Y = varargin{sys_id}.forward(X, part_id);
    end

    function X = adjoint_helper(Y, i)
        sys_id = part_map(i,1);
        part_id = part_map(i,2);
        X = varargin{sys_id}.adjoint(Y, part_id);
    end

    function Y = gather_helper(Y, i)
        sys_id = part_map(i,1);
        part_id = part_map(i,2);
        Y = gathers{sys_id}(Y, part_id);
    end

    function Y = scatter_helper(Y, i)
        sys_id = part_map(i,1);
        part_id = part_map(i,2);
        Y = scatters{sys_id}(Y, part_id);
    end

end

