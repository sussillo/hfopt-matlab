function [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackMRNNUtils(net, varargin)
% function [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackMRNNUtils(net, varargin)
%
% written by David Sussillo (C) 2013
%
% Sometimes we want to reshape more things than just the synaptic
% parameters, for example the costMask or the modMask.  It will give you
% theta, the synaptic and bias parameters, as the default.

optargin = size(varargin,2);

do_theta = false;
do_cost_mask = false;
do_mod_mask = false;

for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
        case 'dotheta'
            do_theta = varargin{i+1};
        case 'docostmask'
            do_cost_mask = varargin{i+1};
        case 'domodmask'
            do_mod_mask = varargin{i+1};
        otherwise
            assert ( false, [' Variable argument ' varargin{i} ' not recognized.']);
    end
end

if ~do_theta && ~do_cost_mask && ~do_mod_mask
    do_theta = true;
end

%nparams = sum([net.layers.nParams]);
nparams_cumsum = cumsum([net.layers.nParams]);
nparams_cumsum = [0 nparams_cumsum];

nlayers = net.nlayers;
nics = net.nICs;
W = cell(1,nlayers);
b = cell(1,nlayers);
for i = 1:nlayers
    layer_start_idx = nparams_cumsum(i)+1;
    layer_stop_idx = nparams_cumsum(i+1);

    npost = net.layers(i).nPost;
    npre = net.layers(i).nPre;
        
    if do_theta
        W_and_b = net.theta(layer_start_idx:layer_stop_idx);
    elseif do_cost_mask
        W_and_b = net.costMask(layer_start_idx:layer_stop_idx);
    elseif do_mod_mask
        W_and_b = net.modMask(layer_start_idx:layer_stop_idx);
    else
        assert ( false, 'Case not implemented yet.');
    end       
    
    if i == 1 
        W{i} = reshape( W_and_b(1:end-nics*npost), npost, npre);
        b{i} = reshape(W_and_b(end-nics*npost+1:end), npost, nics);   
    elseif i == 2
        F = net.layers(i).nFactors;
        V = net.layers(1).nPre;
        N = npost;        
        start_idx = 1; 
        stop_idx = N*F;        
        W{i}.n_W_f = reshape( W_and_b(start_idx:stop_idx), N, F);
        start_idx = stop_idx + 1;
        stop_idx = stop_idx + F*V;
        W{i}.f_W_v = reshape( W_and_b(start_idx:stop_idx), F, V);
        start_idx = stop_idx + 1;
        stop_idx = stop_idx + F*N;
        W{i}.f_W_n = reshape( W_and_b(start_idx:stop_idx), F, N); 
        start_idx = stop_idx + 1;
        stop_idx = stop_idx + N;
        b{i} = W_and_b(start_idx:stop_idx);
    elseif i == 3
        W{i} = reshape( W_and_b(1:end-npost), npost, npre);
        b{i} = W_and_b(end-npost+1:end);   
    end
    
end

% hack cuz first layer is fake input, so using bias as the x(0), for potentially multiple conditions
n_Wru_v = W{1};
n_Wrr_n = W{2};
n_x0_c = b{1};				% xxx, hack that all the initial conditions for the network are in the first layer.
n_bx_1 = b{2};
m_Wzr_n = W{3};
m_bz_1 = b{3};