function eval_network_test = create_eval_network_test2(weight_cost)

eval_network_test = @(net, v_input_, m_target_, training_vs_validation, trial_idx, optional_args, simdata) ...
    bnb_hf_allfun_RLMC(net, v_input_, m_target_, weight_cost, ...
		 [], [], [], training_vs_validation, trial_idx, optional_args, simdata, 1, 0, 0, 0, 0);

end
