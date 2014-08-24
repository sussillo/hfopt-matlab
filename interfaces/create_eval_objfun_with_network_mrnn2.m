function eval_objfun_mrnn2 = create_eval_objfun_with_network_mrnn2(weight_cost)
% Written by David Sussillo (C) 2013
eval_objfun_mrnn2 = @(net, v_input_, m_target_, forward_pass, training_vs_validation, trial_idx, optional_args, simdata) ...
    mrnn_hf_allfun2(net, v_input_, m_target_, weight_cost, ...
    [], [], forward_pass, training_vs_validation, trial_idx, optional_args, simdata, 0, 1, 0, 0, 0);

end
