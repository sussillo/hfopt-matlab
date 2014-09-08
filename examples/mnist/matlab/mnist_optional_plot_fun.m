function plot_stats = mnist_optional_plot_fun(net, simparams, fun, did_objfun_improve_this_iter, f, trial_idxs, forward_pass, ...
    v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
    m_targettrain_T, m_targettrain_s, m_targettest_t, optional_args)

plot_stats = [];

figure(f);
hold off;

TvV_T = 1;
%TvV_V = 2;
%forward_pass = fun.evalNetwork(net, v_inputtest_t, m_targettest_t, TvV_V, 1, {}, 'doparallel', true); % Not content to use mismatch from before CG iter and network update.
%[~,midxs_labels] = max(m_targettest_t);
forward_pass = fun.evalNetwork(net, v_inputtrain_T, m_targettrain_T, TvV_T, 1, {}, 'doparallel', true); % Not content to use mismatch from before CG iter and network update.
[~,midxs_labels] = max(m_targettrain_T);

[~,midxs_outputs] = max(forward_pass{end});

errors = length(find(midxs_labels-midxs_outputs ~= 0));

disp(['# Test errors: ' num2str(errors)]);

do_subplot = true;


%% Visualize input features.


[W, b] = unpackDN(net, net.theta);	% What's the 'u' for?  Keep the same from Martens.

rpidxs = randperm(size(W{1},1));

vW1 = W{1}(rpidxs(1:10),:)';


display_network(vW1, 28);


