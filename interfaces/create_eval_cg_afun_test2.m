function eval_cg_afun_test = create_eval_cg_afun_test2(weight_cost)

eval_cg_afun_test = @(net, v_input_, m_target_, v, lambda, forward_pass, training_vs_validation, trial_idx, optional_args, simdata) ...
    bnb_hf_allfun_RLMC(net, v_input_, m_target_, weight_cost, v, lambda, forward_pass, training_vs_validation, trial_idx, ...
		  optional_args, simdata, 0, 0, 0, 1, 0);

end
