function plot_stats = pathos_optional_plot_fun2(net, simparams, funs, did_objfun_improve_this_iter, fignum, trial_idxs, ...
    forward_pass_T, forward_pass_s, ...				 
    v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
    m_targettrain_T, m_targettrain_s, m_targettest_t, ...
    all_optional_args, all_simdata, all_plot_stats)


% After, each optimization step, the optimizer calls this function.  It
% passes you damned near everything, so you can examine any state that you
% wish.
%
% net - the network structure
% simparams - the structure of parameters that is passed everywhere
% funs - these are the functions you can call to run the network
% did_objfun_improve_this_iter - true|false
% fignum - figure handle
% trial_idxs - the trial indices used in the CG minibatch (I think)
% forward_pass_T, - the entire forward pass
% forward_pass_s - the forward pass used in the minibatch
% v_inputtrain_T - the entire input
% v_inputtrain_s - the input used in the minibatch
% v_inputtest_t  - the validation input
% m_targettrain_T, - the entire target set
% m_targettrain_s - the target set used in the CG minibatch
% m_targettest_t - the target set used in validation
% all_optional_args - strcuture passed around, that is populated by the
%     user defined eval function
% all_simdata - data structure passed around that is populated by the user
% defined functions
% all_plot_stats - a hook for data to be passed back and forth from this
%     function across hf iteration


plot_stats = [];

figure(fignum);
hold off;

[n_Wru_v, n_Wrr_n, m_Wzr_n, n_r0_1, n_br_1, m_bz_1] = unpackRNN(net, net.theta);
W2_n = sum(n_Wrr_n.^2,1);   % sum_i W_ij^2
avg_frob_squared = 0.0;
% Not taking into account capacity for subindices.
this_frob_squared = zeros(1, length(forward_pass_T));
for i = 1:length(forward_pass_T)
    n_dr_t = net.layers(2).derivFunAct(forward_pass_T{i}{1});
    T = size(n_dr_t, 2);
    N = size(n_dr_t, 1);
    
    % 1/T sum_j sum_t r'(t)^2_j sum_i Jij^2
    
    dr2_n = 1.0/T * sum(n_dr_t.^2, 2)'; % sum_t (r'(t)_j)^2
    frob_squared = sum(W2_n .* dr2_n);   % sum_j sum_t (r'(t)_j)^2 sum_i W_ij^2
    this_frob_squared(i) = frob_squared;     % Average over T to get the squared Frobenius norm.    
    
    avg_frob_squared = avg_frob_squared + this_frob_squared(i);
end
avg_frob_squared = avg_frob_squared / length(forward_pass_T);
disp(['<Frobenius norm squared>: ' num2str(avg_frob_squared) '.']);
    


for i = 1:length(forward_pass_s)
    if ( i > 4 )
        continue;
    end
    subplot(4,2,i);

    
    idx = randi(length(forward_pass_s));
    
    Tp = size(forward_pass_s{idx}{3},2);    
    plot(forward_pass_s{idx}{3}', '-xb');
    hold on;
    stem(Tp, forward_pass_s{idx}{3}(end), 'bx', 'linewidth', 2);
    stem(m_targettrain_s{idx}(1:end), 'rx', 'linewidth', 2);
    if ( size(v_inputtrain_s{idx},1) > 1 )
        stem(1:Tp, v_inputtrain_s{idx}(2,:) .* v_inputtrain_s{idx}(1,:), 'kx', 'linewidth', 2)
    end
    axis tight;
    ylim([0 1]);
    hold off;
    
    subplot(4,2,4+i);
    
    
    plot((forward_pass_s{idx}{1}(1:5,:) + repmat(linspace(0,8,5)', 1, Tp))', 'b');
    axis tight;
end


%const_input_vals = [0.5; 0.0];  % Open question as to whether the "constant" input should be the input mean.
%nfps = 1;
%init_eps = 0.01;
%fun_tol = 8e-7;
%[fp_struct_avg, ~] = find_many_fixed(net, nfps, forward_pass_s{1}{1}(:,50), init_eps, 100.0, fun_tol, 'constinput', const_input_vals);


%[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_br_1, m_bz_1] = unpackRNN(net, net.theta);
%[V,D] = eig(n_Wrr_n);
%if ~isempty(fp_struct_avg)
if 1
    subplot(4,2,8);
    %plot(fp_struct_avg(1).eigenValues, 'x');
    
    fjacs = zeros(1,T);
    avg_jac = zeros(N,N);
    for t = 1:T    
        jac = n_Wrr_n .* repmat(n_dr_t(:,t)', N, 1);
        avg_jac = avg_jac + jac;
        fjacs(t) = norm(jac, 'fro'); 
    end
    avg_jac = avg_jac / T;
    
    D = eig(avg_jac);
    plot(D, 'x');
    
    axis square;
end