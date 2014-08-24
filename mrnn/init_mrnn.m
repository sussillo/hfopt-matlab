function net = init_mrnn(layer_sizes, layer_types, g, obj_fun_type, varargin)
% function net = init_mrnn(layer_sizes, layer_types, g, obj_fun_type, varargin)
%
% % Written by David Sussillo (C) 2013
%
% Keep the same form, but for RNN these are just two layers deep.  I->H H->O
%
% layer_sizes are the input, first hidden layer output, second hidden layer output, ..., final hidden layer output.
%
% In writing this Matlab version of the RNN, I am going to be thinking of an input driven device, not dissimilar from
% a DBN.  So I can feed it batch inputs (temporally ordered, of course), iteratively compute the forward pass, but
% otherwise use the HF mini-batch algorithm on it.
%
% A layer is respect to the weights, so input -> weights -> recurrent  ( layer 1 )
%                                       recurrent -> weights -> recurrent  ( layer 2 )
%                                       recurrent -> weights -> output  ( layer 3 )
%

% Trying to use layered struct here is stupid.  Basically, it's a square peg and round hole, but it does save a
% little bit of code.  Should be rewritten. -DCS:2011/09/14

nlayers = length(layer_sizes)-1;	% the first layer size is the input.
assert ( nlayers == 3, 'RNN' );  % middle layer is recurrent weights.  I have no idea if this is a good idea or not.
assert ( length(g) == 4, 'Should be size 3, (I->N)input via biases (N->N)rec in tensor, (N->M) output, (I->F) input to tensor in factors.');
assert ( length(layer_types) == 3, 'Should be size 3, (I->N) (N->N) (N->M).');

% These are for traditional biases.
do_learn_biases = true;
do_init_state_biases_random = false;  % Should be an optional parameter to the function, probably.  They start at zero, they may stay there.

% I've come to realize these are simply totally different than biases.
do_learn_state_init = true;  % this is n_x0_c
do_init_state_init_random = false;  % this is n_x0_c

% Both of these masks have the same 'shape' as the theta vector of parameters, and can be formatted into matrices or
% vectors using the unpackRNNUtils.m function.
cost_mask_layer_fac = ones(1000,1);		% could break
mod_mask_layer_fac = ones(1000,1);	% the modification mask.  This is just a way to have matrices of params, but
% really have sparseness or change what's learned.  Similar, but
% fundamentally in purpose from a cost mask.  This excludes these
% 'parameters' from any calculation of the cost, or any modification
% whatsoever, if the mod mask is 0.  If 1, it's like normal.

bias_scale = 1.0;

% Layer by layer sparsity parameters
tau = 1.0;				% default is discrete RNN, when tau = dt
dt = 1.0;				% default is discrete RNN, when tau = dt
mu = 1.0;				% no idea what's reasonable for this value.
nics = 1;				% number of initial conditions to learn, based on some type of external condition
net_noise_sigma = 0.0;			% gaussian noise standard dev, multiplied by sqrt of dt, which for discrete models is 1

% Not sure what the right default is. 1 is natural, but it means that n_Wrr_n is a scaled outer product.  That doesn't seem powerful.
% Another natural option is N, the number of hidden units, I guess.
F = layer_sizes(2);   

optargin = size(varargin,2);
trans_fun_params = [];
for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
        case 'numconn'
            assert (false, 'numconn is not used in an MRNN');
        case 'wcfacbylayer'
            assert ( false, 'this was always the cost mask! Sorry! ');
        case {'cmfacbylayer', 'costmaskfacbylayer'}
            cost_mask_layer_fac = varargin{i+1};
            assert(length(cost_mask_layer_fac) == 3, 'Should be size 3, (I->N) (N->N) (N->M).');
        case 'modmaskbylayer'
            mod_mask_layer_fac = varargin{i+1};
        case 'tau'
            tau = varargin{i+1};
        case 'dt'
            dt = varargin{i+1};
        case 'mu'
            mu = varargin{i+1};
        case 'netnoisesigma'
            net_noise_sigma = varargin{i+1};
        case 'nics'
            nics = varargin{i+1};
        case 'dolearnstateinit'
            do_learn_state_init = varargin{i+1};
        case 'doinitstateinitrandom'
            do_init_state_init_random = varargin{i+1};
        case {'dolearnbiases', 'dobiases'}
            do_learn_biases = varargin{i+1};
        case 'doinitstatebiasesrandom'		% this is for traditional biases (not output!)
            do_init_state_biases_random = varargin{i+1};
        case 'biasscale'
            bias_scale = varargin{i+1};
        case 'doinitbiasesrandom'
            assert ( false, 'Initing the biases randomly is only for state, so previous label is misleading. Sorry!');
        case 'transfunparams'
            trans_fun_params = varargin{i+1};  %e.g. stanh takes a single alpha parameter, send as an array
        case 'nfactors'
            F = varargin{i+1};  % This is the number of tensor product factors in n_Wrr_n, see Sutskever et al. ICML 2011            
        otherwise
            assert( false, ['Don''t recognize ' varargin{i} '.']);
    end
end

V = layer_sizes(1);
N = layer_sizes(2);
M = layer_sizes(4);        

npres = layer_sizes(1:end-1);
nposts = layer_sizes(2:end);

assert ( dt/tau <= 1.0, 'Dude...');

net.type = 'MRNN';
net.tau = tau;
net.dt = dt;
net.mu = mu;
net.nlayers = nlayers;
W = cell(1,nlayers);
for i = 1:nlayers
    npre = npres(i);
    npost = nposts(i);
    
    
    if i == 1 || i == 3 % Like I said, square peg - round hole.
        W{i} = g(i) * randn(npost,npre)/sqrt(npre);
    elseif i == 2  % Like I said, square peg - round hole.
        % n_Wrr_n = h_W_f * f_diag( f_W_v v_input_1)_f * f_W_h
        
        layers(i).nFactors = F;
        Wrr.n_W_f = sqrt(g(2)) * randn(N,F) / sqrt(F);  % hiddens <- factors
        Wrr.f_W_v = g(4) * randn(F,V) / sqrt(V); % (factors * inputs ) 
        Wrr.f_W_n = sqrt(g(2)) * randn(F,N) / sqrt(N); % factors <- hiddens
        W{i} = Wrr;        
        
        % For the recurrent network we'll set things exactly.  The above heuristic is OK for feed-forward connections, but let's be precise for J.
        %warning('We should really figure out how to scale the internal weight tensor appropriately, presumably for a random unit norm input vector');
        %D = eig(W{i});
        %max_real_eig = max(max(real(D)));
        %W{i} = W{i} / max_real_eig * g(i);
        %disp(num2str(g(i)/max_real_eig));  % This should always be close to 1 unless something is very wrong.        
    end
    layers(i).nPre = npre;
    layers(i).nPost = npost;
    layers(i).type = layer_types{i};
    
    bias{i} = zeros(npost,1);
    if ( do_init_state_biases_random && i < 3)
        bias{i} = bias_scale * 2.0*(rand(npost,1)-0.5);
    end
    
    if strcmpi(layers(i).type, 'logistic')
        layers(i).transFun = @(x) 1.0 ./ (1.0 + exp(-x)); %#ok<*AGROW>
        layers(i).derivFunAct = @(y) y.*(1.0-y);
        layers(i).invTransFun = @(y) log( y ./ (1-y) );
    elseif strcmpi(layers(i).type, 'softmax')
        layers(i).transFun = @mysoftmax;
        layers(i).derivFunAct = @softmax_derivs_act;
        assert ( i == nlayers, 'Not sure what happens if this isn''t ath the output, you should check it.');
        layers(i).invTransFun = @not_implemented;
    elseif strcmpi(layers(i).type, 'linear')
        layers(i).transFun = @identity;
        layers(i).invTransFun = @identity;
        layers(i).derivFunAct = @(y) ones(size(y));
    elseif strcmpi(layers(i).type, 'exp')
        layers(i).transFun = @exp;
        layers(i).invTransFun = @log;
        layers(i).derivFunAct = @(r) r;
    elseif strcmpi(layers(i).type, 'rectlinear')
        layers(i).transFun = @(x)  (x > 0) .* x;
        layers(i).invTransFun = @(r) assert(false, 'no inverse');
        layers(i).derivFunAct = @(r) (r > 0) .* ones(size(r));
    elseif strcmpi(layers(i).type, 'tanh')
        layers(i).transFun = @tanh;
        layers(i).derivFunAct = @(r) 1.0-r.^2;
        layers(i).invTransFun = @atanh;
        layers(i).derivInvTransFun = @(r) 1.0 ./ (1.0 - r.^2);
    elseif strcmpi(layers(i).type, 'recttanh')
        layers(i).transFun = @(x) (x > 0 ) .* tanh(x);
        layers(i).invTransFun = @(r) assert(false, 'no inverse');
        layers(i).derivFunAct = @(r) (r > 0) .* (1.0 - r.^2);
    elseif strcmpi(layers(i).type, 'stanh')
        alpha = trans_fun_params(1);
        layers(i).transFun = @(x) (1.0/alpha) * tanh(alpha*x);
        layers(i).derivFunAct = @(r) 1.0-(alpha*r).^2;
        layers(i).invTransFun = @(r) (1.0/alpha) * atanh(alpha*r);
        layers(i).derivInvTransFun = @(r) 0;
    elseif strcmpi(layers(i).type, 'LHK')  % Larry, Haim, Kanaka
        R0 = trans_fun_params(1);
        layers(i).transFun = @(x)  (R0.*tanh(x./R0) .* (x <= 0.0 )) + ((2.0-R0).*tanh(x./(2.0-R0)) .* ( x > 0.0));
        layers(i).derivFunAct = @(r)  ((1 - r.^2./R0.^2) .* (r <= 0.0 )) + ((1 - r.^2./(2.0-R0).^2) .* (r > 0.0));
        layers(i).invTransFun = @(r) assert(false, 'no inverse yet.');
        layers(i).derivInvTransFun = @(r) 0;
        
    elseif strcmpi(layers(i).type, 'rectstanh')
        alpha = trans_fun_params(1);
        layers(i).transFun = @(x) (x > 0) .* ((1.0/alpha) * tanh(alpha*x));
        layers(i).derivFunAct = @(r) (r > 0) .* (1.0-(alpha*r).^2);
        layers(i).invTransFun = @(r) assert(false, 'no inverse');
        layers(i).derivInvTransFun = @(r) 1.0 ./ (1.0 - r.^2);
    elseif strcmpi(layers(i).type, 'pow')
        pow = trans_fun_params(1);
        % Needs to be rewritten
        layers(i).transFun = @(x) (x > 0) .* x.^pow - (x <= 0).*((-x).^pow);
        layers(i).derivFun = @(x) ( ((x>0) .* pow .* x.^(pow-1)) + ( (x<=0) .* pow .* (-x).^(pow-1)));
        layers(i).derivFunAct = @(r) (0);
    else
        assert ( false, 'This tranfer function is not implmeneted yet.');
    end
    
    if i == 1
        layers(i).nParams = V * N + nics*N; % weights + nics * biases, fucking ic biases are in layer 1! -DCS:2011/10/25
    elseif i == 2
        layers(i).nParams = N*F + F*V + F*N + N;   % This is all the Ws in "Wrf * diag( Wfv * v ) * Wfr", plus the biases
    elseif i == 3
        layers(i).nParams = N * M + M; % weights + biases
    else 
        assert ( false, 'Fucked');
    end
end


% Create the state initialization.
n_x0s_c = zeros(layers(2).nPre,nics);
if do_init_state_init_random
    n_x0s_c = 2.0*(rand(layers(2).nPre,nics)-0.5);
end

%n_x0s_c, checking cuz gets reshaped
net.doLearnStateInit = do_learn_state_init;
net.doLearnBiases = do_learn_biases;
net.nICs = nics;
net.layers = layers;
net.init.g = g;
theta = packMRNN(net, W{1}, W{2}, W{3}, n_x0s_c, bias{2}, bias{3});
net.theta = theta;
net.originalTheta = net.theta;		% very useful sometimes for analysis.
net.originalX0s = n_x0s_c;
net.noiseSigma = net_noise_sigma * sqrt(net.dt);

% Create the cost mask for weight cost on weights only, not biases.
nparams = sum([net.layers.nParams]);
nparams_cumsum = cumsum([net.layers.nParams]);
nparams_cumsum = [0 nparams_cumsum];
cost_mask = zeros(nparams, 1); % Default is that all parameters have no cost associated with them.
mod_mask = ones(nparams,1);    % Default is that all parameters are actually parameters.
for i = 1:nlayers
    %npre = net.layers(i).nPre;
    npost = net.layers(i).nPost;
    
    mask_start_idx = nparams_cumsum(i)+1;
    mask_stop_idx = nparams_cumsum(i+1)-npost;
    mask_size = length(mask_start_idx:mask_stop_idx);
    
    cost_mask(mask_start_idx:mask_stop_idx) = cost_mask_layer_fac(i) * ones(mask_size,1);
    mod_mask(mask_start_idx:mask_stop_idx) = mod_mask_layer_fac(i) * ones(mask_size,1);
    
    net.layers(i).thetaIdxs = mask_start_idx:(mask_stop_idx+npost);
    net.layers(i).thetaWIdxs = mask_start_idx:mask_stop_idx; % xxx, wrong, needs to be fixed, but whole thing does so not
    % fixing.  -DCS:2011/10/25
    net.layers(i).thetaBIdxs = (mask_stop_idx+1):(mask_stop_idx+npost);
end

net.costMask = cost_mask;
net.modMask = mod_mask;
% One should xor the heaviside of these masks to make sure the xor is always 1, and give an error otherwise -DCS:2012/03/15.
% xxx We should continue this.

if (strcmpi(obj_fun_type, 'cross-entropy') )
    net.objectiveFunction = 'cross-entropy';
elseif (strcmpi(obj_fun_type, 'sum-of-squares') )
    net.objectiveFunction = 'sum-of-squares';
elseif (strcmpi(obj_fun_type, 'nll-poisson') )
    net.objectiveFUnction = 'nll-poisson';
else
    assert ( false, 'Objective function not implemented yet.');
end


net.hasCanonicalLink = false;
if ( strcmp(net.objectiveFunction, 'cross-entropy') && ...
        ( strcmp(net.layers(end).type, 'logistic')  || ...
        strcmp(net.layers(end).type, 'softmax') ))
    
    net.hasCanonicalLink = true;
elseif ( strcmp(net.objectiveFunction, 'sum-of-squares') && ...
        strcmp(net.layers(end).type, 'linear') )
    net.hasCanonicalLink = true;
elseif ( strcmp(net.objectiveFunction, 'nll-poisson') && ...
        strcmp(net.layers(end).type, 'exp') )
    net.hasCanonicalLink = true;

end