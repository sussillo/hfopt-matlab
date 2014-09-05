function net = init_dn(layer_sizes, layer_types, g, obj_fun_type, varargin)
% Init single hidden layer network.  Don't bother with anything except the weights and biases for now.
%
% layer_sizes are the input, first hidden layer output, second hidden layer output, ..., final hidden layer output.

nlayers = length(layer_sizes)-1;	% the first layer size is the input.

% A layer is respect to the weights, so input -> weights -> output  ( layer 1 )
%                                       input -> weights -> output  ( layer 2 )
%                                       input -> weights -> output  ( layer 3 )
%                                       input -> weights -> output  ( layer 4 )
% is a four layer network according to this definition, cuz there are four transformations.

%assert ( nlayers > 1, 'More layers!' );

wc_layer_fac = ones(1000,1);		% could break
numconn = 15;

% Layer by layer sparsity parameters
beta = [];
rho = [];

optargin = size(varargin,2);
for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
%         case 'numconn'
%             numconn = varargin{i+1};
        case 'wcfacbylayer'
            wc_layer_fac = varargin{i+1};
        case 'beta'
            beta = varargin{i+1};
        case 'rho'
            rho = varargin{i+1};
    end
end

npres = layer_sizes(1:end-1);
nposts = layer_sizes(2:end);
net.nlayers = nlayers;
W = cell(1,nlayers);
layer = struct;
layers = repmat(layer, 1, nlayers);
for i = 1:nlayers
    npre = npres(i);
    npost = nposts(i);
    
    W{i} = g(i) * randn(npost,npre)/sqrt(npre);
    
%     for j = 1:npost
%         if isinf(numconn)
%             idxs = 1:npre;
%             W{i}(j,idxs) = (2.0*(rand(npre,1)-0.5)) * sqrt(6)/sqrt(npre+npost);
%         else
%             assert( false, 'Eat shit and die.');
%             idxs = ceil(layer_sizes(i)*rand(1,numconn));
%             W{i}(j,idxs) = randn(numconn,1);
%         end
%         n = norm(W{i}(j,:));
%         W{i}(j,idxs) = W{i}(j,idxs) * g(i);
%     end
%     
  
    layers(i).nPre = npre;
    layers(i).nPost = npost;
    layers(i).type = layer_types{i};
    
    bias{i} = zeros(npost,1);
    %if ( strcmpi(layers(i).type, 'tanh') )
    %bias{i} = 0.5 * ones(npost,1);
    %end
    
    if strcmpi(layers(i).type, 'logistic')
        layers(i).transFun = @(x) 1.0 ./ (1.0 + exp(-x));
        layers(i).derivFunAct = @(y) y.*(1.0-y);
        %layers(i).deriv2FunAct = @(y) y - 3*y.^2 + 2*y.^3;        
        layers(i).deriv2FunAct = @(y) y .* ( 1 - y .* ( -3  + 2*y));  % A lot faster
    elseif strcmpi(layers(i).type, 'softmax')
        layers(i).transFun = @mysoftmax;
        layers(i).derivFunAct = @(y) diag(y) - y*y';  % Will break for multiple examples, cuz 3D.
    elseif strcmpi(layers(i).type, 'linear')
        layers(i).transFun = @(x) x;
        layers(i).derivFunAct = @(y) ones(size(y));
        layers(i).deriv2FunAct = @(y) zeros(size(y));
    elseif strcmpi(layers(i).type, 'tanh')
        layers(i).transFun = @tanh;
        layers(i).derivFunAct = @(y) 1.0-y.^2;
        layers(i).deriv2FunAct = @(y) -2.0*y .* (1.0 - y.^2);
    elseif strcmpi(layers(i).type, 'rectlinear')
        layers(i).transFun = @(x) (x > 0) .* x;
        layers(i).derivFunAct = @(y) (y > 0);
    elseif strcmpi(layers(i).type, 'softplus')
        layers(i).transFun = @(x) log(1 + exp(x));
        layers(i).derivFunAct = @(y) ((exp(y)-1)./exp(y));

    else
        assert ( false, 'This tranfer function is not implmeneted yet.');
    end
    
    if ( ~isempty(beta) && ~isempty(rho) )
        layers(i).beta = beta(i);
        layers(i).rho = rho(i);
    end
    
    
    layers(i).nParams = npre * npost + npost; % weights + biases
end



net.layers = layers;
net.init.g = g;
net.theta = packDN(net, W, bias);


% Create the cost mask for weight cost on weights only, not biases.
nparams = sum([net.layers.nParams]);
nparams_cumsum = cumsum([net.layers.nParams]);
nparams_cumsum = [0 nparams_cumsum];
cost_mask = zeros(nparams, 1);
for i = 1:nlayers
    npre = net.layers(i).nPre;
    npost = net.layers(i).nPost;
    
    mask_start_idx = nparams_cumsum(i)+1;
    mask_stop_idx = nparams_cumsum(i+1)-npost;
    mask_size = length(mask_start_idx:mask_stop_idx);
    
    cost_mask(mask_start_idx:mask_stop_idx) = wc_layer_fac(i) * ones(mask_size,1);
    
    net.layers(i).thetaIdxs = mask_start_idx:(mask_stop_idx+npost);
    net.layers(i).thetaWIdxs = mask_start_idx:mask_stop_idx;
    net.layers(i).thetaBIdxs = (mask_stop_idx+1):(mask_stop_idx+npost);
end

net.costMask = cost_mask;

if (strcmpi(obj_fun_type, 'cross-entropy') )
    net.objectiveFunction = 'cross-entropy';
elseif (strcmpi(obj_fun_type, 'sum-of-squares') )
    net.objectiveFunction = 'sum-of-squares';
else
    assert ( false, 'Objective function not implemented yet.');
end


net.hasCanonicalLink = false;
if ( strcmp(net.objectiveFunction, 'cross-entropy') & ...
        ( strcmp(net.layers(end).type, 'logistic') || strcmp(net.layers(end).type, 'softmax')))
    net.hasCanonicalLink = true;
elseif ( strcmp(net.objectiveFunction, 'sum-of-squares') & ...
        strcmp(net.layers(end).type, 'linear') )
    net.hasCanonicalLink = true;
end