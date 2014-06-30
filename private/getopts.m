function opts = getopts(defaults, varargin)
%GETOPTS    Parse key-value pairs in varargin into a struct
%   opts = GETOPTS(defaults, param1, value1, param2, value2, ...) will grab
%   an arbitrary number of parameter/value pairs and return them in the
%   struct opts. The parameter names must already be fields of the struct
%   defaults or an error will be generated. The only exception to this rule
%   is the 'opts' parameter name (see below).
%
%   opts = GETOPTS(defaults, 'opts', opts) will update the parameter/value
%   pairs stored in defaults with the ones stored in opts and return the
%   resulting struct. Fields of opts must be existing fields in defaults or
%   an error will result. The presence of the 'opts' parameter takes
%   precedence over the standard parameter/value pairs, so if 'opts' is a
%   parameter passed into GETOPTS, then the other parameter/value pairs
%   are ignored.

if mod(length(varargin),2)
    error('getopts:num_of_arguments', ...
          'arguments must come in parameter/value pairs');
end

% compute number of parameter/value pairs
num_opts = length(varargin)/2;

% copy in defaults
opts = defaults;

for i=1:2:(num_opts*2)
    key = varargin{i};
    value = varargin{i+1};
    if isequal(key, 'opts')
        % parse the parameters in opts
        params = fieldnames(value);
        opts = defaults; % ignore previous changes
        for j=1:numel(params)
            if isfield(defaults, params{j})
                opts.(params{j}) = value.(params{j});
            else
                error('getopts:unknown_parameter', ...
                      'unknown parameter %s', key);
            end
        end
        % exit the function
        return;
    end
    if isfield(defaults, key)
        opts.(key) = value;
    else
        error('getopts:unknown_parameter', 'unknown parameter %s', key);
    end
end

end

