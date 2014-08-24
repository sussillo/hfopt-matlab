function eval_preconditioner_test = create_eval_preconditioner_test2(weight_cost)

eval_preconditioner_test = @(net, v_input_, m_target_, lambda, training_vs_validation, trial_idx, optional_args, simdata) ...
    bnb_hf_allfun(net, v_input_, m_target_, weight_cost, ...
    [], lambda, [], training_vs_validation, trial_idx, optional_args, simdata, 0, 0, 0, 0, 1);

end
