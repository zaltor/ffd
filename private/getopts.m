function opts = getopts(opts, varargin)
% opts = getopts(opts, ...)
% Update the opts structure with variables
%
% e.g. opts = getopts(opts, 'key', 'value', 'key2', 'value2');

if mod(length(varargin),2)
    error('getopts:num_of_arguments', 'options in the parameters must come in the form of key value pairs');
end

num_opts = length(varargin)/2;

for i=1:2:(num_opts*2)
    key = varargin{i};
    value = varargin{i+1};
    if isfield(opts, key)
        opts.(key) = value;
    else
        error('getopts:unknown_opt', 'unknown key %s', key);
    end
end

end

