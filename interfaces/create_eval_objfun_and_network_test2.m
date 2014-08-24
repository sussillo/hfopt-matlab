function eval_objfun_and_network_test2 = create_eval_objfun_and_network_test2(weight_cost)

eval_objfun_and_network_test2 = @(net, v_input_, m_target_, training_vs_validation, trial_idx, optional_args, simdata) ...
    bnb_hf_allfun_RL(net, v_input_, m_target_, weight_cost, ...
    [], [], [], training_vs_validation, trial_idx, optional_args, simdata, 1, 1, 0, 0, 0);

end
