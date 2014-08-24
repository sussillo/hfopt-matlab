function [W,b] = unpackDN(net, theta)
% NOTE that net.theta IS NOT USED.  If you want to use this function with net.theta, then pass it as the second
% parameter!.  The way it's written, it accomodates the R{} technique for exact computation of the Hessian. 

%nparams = sum([net.layers.nParams]);
nparams_cumsum = cumsum([net.layers.nParams]);
nparams_cumsum = [0 nparams_cumsum];

nlayers = net.nlayers;
for i = 1:nlayers
    layer_start_idx = nparams_cumsum(i)+1;
    layer_stop_idx = nparams_cumsum(i+1);

    npost = net.layers(i).nPost;
    npre = net.layers(i).nPre;
    
    W_and_b  = theta(layer_start_idx:layer_stop_idx);
    W{i} = reshape( W_and_b(1:end-npost), npost, npre); %#ok<AGROW>
    b{i} = W_and_b(end-npost+1:end); %#ok<AGROW>
end
