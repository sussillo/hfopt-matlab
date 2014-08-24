function theta = packDBN(net, Wc, biasc)
% This function is appropriate for a FFN with a single hidden layer.  Obviously, the variable names are wrt to the
% actual weights, but any derivative products during learning can be packed / unpacked in this way.
%
% Note that (obviously) net isn't modified or returned.  This accomodates the R{} technique implementation.

nparams = sum([net.layers.nParams]);
nparams_cumsum = cumsum([net.layers.nParams]);
nparams_cumsum = [0 nparams_cumsum];

theta = zeros(nparams, 1);
nlayers = net.nlayers;
for i = 1:nlayers
    layer_start_idx = nparams_cumsum(i)+1;
    layer_stop_idx = nparams_cumsum(i+1);
    theta(layer_start_idx:layer_stop_idx) = [ vec(Wc{i}) ; vec(biasc{i}) ];
end
