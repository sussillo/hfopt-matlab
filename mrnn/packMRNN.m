function theta = packMRNN(net, n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1)
% function theta = packMRNN(net, n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1)
%
% Written by David Sussillo (C) 2013
%
% This function is appropriate for a FFN with a single hidden layer.  Obviously, the variable names are wrt to the
% actual weights, but any derivative products during learning can be packed / unpacked in this way.
%
% Note that (obviously) net isn't modified or returned.  This accomodates the R{} technique implementation.
%
% hack cuz first layer is fake input, so using bias as the x(0), for multiple conditoins

nparams = sum([net.layers.nParams]);
nparams_cumsum = cumsum([net.layers.nParams]);
nparams_cumsum = [0 nparams_cumsum];

W{1} = n_Wru_v;
%N = size(n_Wru_v,1);			% hack cuz first layer is fake input
bias{1} = n_x0_c;			% this is n x nics, xxx, should be reimplemented
W{2} = n_Wrr_n;
bias{2} = n_bx_1;
W{3} = m_Wzr_n;
bias{3} = m_bz_1;

theta = zeros(nparams, 1);
nlayers = net.nlayers;
for i = 1:nlayers
    layer_start_idx = nparams_cumsum(i)+1;
    layer_stop_idx = nparams_cumsum(i+1);
    if i == 1 || i == 3
        theta(layer_start_idx:layer_stop_idx) = [ vec(W{i}) ; vec(bias{i}) ];
    elseif i == 2
        theta(layer_start_idx:layer_stop_idx) = [ vec(W{i}.n_W_f) ; vec(W{i}.f_W_v); vec(W{i}.f_W_n); vec(bias{i}) ];
    else
        assert ( false, 'Fucked');
    end                
end
