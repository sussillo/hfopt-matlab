function [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, theta)
% NOTE that net.theta IS NOT USED.  If you want to use this function with net.theta, then pass it as the second
% parameter!.  The way it's written, it accomodates the R{} technique for exact computation of the Hessian. 


nparams = sum([net.layers.nParams]);
nparams_cumsum = cumsum([net.layers.nParams]);
nparams_cumsum = [0 nparams_cumsum];

nlayers = net.nlayers;
nics = net.nICs;
for i = 1:nlayers
    layer_start_idx = nparams_cumsum(i)+1;
    layer_stop_idx = nparams_cumsum(i+1);

    npost = net.layers(i).nPost;
    npre = net.layers(i).nPre;
    
    W_and_b  = theta(layer_start_idx:layer_stop_idx);
    if i == 1 
        W{i} = reshape( W_and_b(1:end-nics*npost), npost, npre);
        b{i} = reshape(W_and_b(end-nics*npost+1:end), npost, nics);   
    else
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