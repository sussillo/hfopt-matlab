function eval_network_mrnn = create_eval_network_mrnn2(weight_cost)
% Written by David Sussillo (C) 2013
eval_network_mrnn = @(net, v_input_, m_target_, training_vs_validation, trial_idx, optional_args, simdata) ...
    mrnn_hf_allfun2(net, v_input_, m_target_, weight_cost, ...
		 [], [], [], training_vs_validation, trial_idx, optional_args, simdata, 1, 0, 0, 0, 0);

end
