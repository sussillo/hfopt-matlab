function eval_objfun_and_network_mrnn2 = create_eval_objfun_and_network_mrnn2(weight_cost)

eval_objfun_and_network_mrnn2 = @(net, v_input_, m_target_, training_vs_validation, trial_idx, optional_args, simdata) ...
    mrnn_hf_allfun2(net, v_input_, m_target_, weight_cost, ...
    [], [], [], training_vs_validation, trial_idx, optional_args, simdata, 1, 1, 0, 0, 0);

end
