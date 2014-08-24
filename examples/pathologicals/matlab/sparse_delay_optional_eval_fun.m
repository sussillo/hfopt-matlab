function optional_params = sparse_delay_optional_eval_fun(net, fun, did_objfun_improve_this_iter, f, forward_pass_s, ...
						  v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
						  m_targettrain_T, m_targettrain_s, m_targettest_t)


eval_network = funs.evalNetwork;
forward_pass = eval_network(net, v_inputtrain_T, m_targettrain_T, {[]});
n_rsum_t = zeros(size(forward_pass{1}{1}));

ntrials = length(forward_pass);
for i = 1:ntrials
    forward_pass_r{i} = forward_pass{i}{1};       
    n_rsum_t = n_rsum_t + forward_pass_r{i};  
end
n_rhoh_1 = mean(n_rsum_t / ntrials,2);
disp(['Mean: ' num2str(mean(n_rhoh_1))]);
optional_params{1} = n_rhoh_1;

