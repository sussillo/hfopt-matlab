function plot_stats = pm_maze_plotfun(net, simparams, fun, did_objfun_improve_this_iter, f, trial_idxs, forward_pass_T, forward_pass_s, ...
    v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
    m_targettrain_T, m_targettrain_s, m_targettest_t, all_optional_args, all_simdata, all_plot_stats)

plot_stats = [];

figure(f);
hold off;

T = length(v_inputtrain_T);
TvV_T = 1;
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_br_1, m_bz_1] = unpackRNN(net, net.theta);
out_trans_fun = net.layers(3).transFun;
% 
% if simparams.doPassValidationTrialsWithTraining
%     % Have to execute this pass to evaluate the objective function.
%     validation_trial_idxs = simparams.validationTrialIdxs;
%     v_inputtrain_valid = v_inputtrain_T(validation_trial_idxs);
%     m_targettrain_valid = m_targettrain_T(validation_trial_idxs);
%     % TvV is T here cuz isn't true validation
%     % all_optional_args and all_simdata are referenced vis TvV_T and the trial indices passed (validation_trial_idxs)
%     package = fun.evalObjfun(net, v_inputtrain_valid, m_targettrain_valid, TvV_T, validation_trial_idxs, all_optional_args, all_simdata, 'doparallel', true);
%     objfun_validation = package{1}; % DOn't care about simdata, cuz it's not passed back.
%     disp(['Validation Error: ' num2str(objfun_validation) ]);
% end
RNN_FP_R_IDX = 1;
%RNN_FP_Z_IDX = 3;
rptidxs = randperm(length(trial_idxs));
plot_idx = 1; %rpidxs(1);
true_plot_idx = rptidxs(1);


r_hidden_t = forward_pass_T{true_plot_idx}{RNN_FP_R_IDX};
tsize = size(r_hidden_t,2);
r_output_t = out_trans_fun( m_Wzr_n * r_hidden_t + repmat(m_bz_1, 1, tsize) ); % a little parallelism


nfactors = simparams.nFactors;

N = size(n_Wru_v,1);
rnet_idxs = 1:N-nfactors;
rnet_mat = n_Wrr_n(rnet_idxs,rnet_idxs);
disp(['Frobenius norm of J: ' num2str(norm(rnet_mat, 'fro')) '.']);
disp(['Trace of J: ' num2str(trace(rnet_mat)) '.']);


subplot(3,3,3)
ntoplot = 8;
%I = size(m_Wzr_n,1);
[N,ntime] = size(r_hidden_t);
rpnidxs = randperm(N);
rpnidxs = rpnidxs(1:ntoplot);
offsets = repmat([1:ntoplot]'*0.5, 1, ntime);
r_hidden_t_offseted = r_hidden_t(rpnidxs,:) + offsets;
plot(r_hidden_t_offseted');
title('RNN Hidden');
axis tight;

subplot(3,3,2)
%imagesc((r_output_T{plot_idx} - m_targettrain_T{true_plot_idx}).^2);
%colorbar;
%title('RNN Output Residuals');
RP = randn(3,N);
rp_ics = RP*n_x0_c;
plot3(rp_ics(1,:), rp_ics(2,:), rp_ics(3,:), 'kx');
axis tight;




% 
% subplot(3,3,3)
% if simparams.doPassValidationTrialsWithTraining
%     hold on;
%     plot(simparams.HFIter, objfun_validation, 'rx', 'linewidth', 4); axis tight;
%     hold off;
%     title('Validation Error');
% else
%    % imagesc(abs(corrcoef(f_pre_T')));
%    1;
%     %colorbar;
%     %title('Abs Predictor Correlation Coefficients');
% end


ntimes = size(m_targettrain_T{true_plot_idx},2);
dt = net.dt;
time = (dt:dt:ntimes*dt) * 1000; % ms is the best time scale

subplot(3,3,8)
imagesc(time, [], [m_targettrain_T{true_plot_idx}]);
title('Target');
%colorbar;
subplot(3,3,5)
imagesc(time, [], r_output_t);
title('RNN Output');
%colorbar;
subplot(3,3,6)
dumb_scale = 4.0;
imagesc(time, [], (m_targettrain_T{true_plot_idx} + dumb_scale*r_output_t)/2.0);
title('Averaged');
%colorbar


f_pre_T = r_hidden_t(end-nfactors+1:end,:);

% f_pre_var = var(f_pre_T, [], 2);
% f_pre_mean = mean(f_pre_T,2);

thing1 = min(f_pre_T');
thing2 = max(f_pre_T');
offset = max(abs(thing2 - thing1));

%disp(['Means of predictor units: ' num2str(f_pre_mean') '.' ])
%disp(['Vars of predictor units: ' num2str(f_pre_var') '.' ])
%norm_pres_reg_weight = net.normedPreWeights.weight;
%norm_pres_reg_dv = net.normedPreWeights.desiredValue;
norm_pres_reg_mask = logical(net.normedPreWeights.mask);
%N_norm_pres_reg = length(find(norm_pres_reg_mask));
m_Wrr_n = n_Wrr_n(norm_pres_reg_mask,:);
[~, l_prenorms_m] = normify(m_Wrr_n');
disp(num2str(l_prenorms_m));
subplot(1,3,1);
for i = 1:nfactors
    plot(time, f_pre_T(i,:)+(i-1)*offset, 'r', 'linewidth', 2);  % This last will be the predictioned single factor in the pm_as_sparse_dyn
    hold on;
end
hold off;
axis tight;

% Visualize the extractor predictions
%extractors = m_Wzr_n(:,end-nfactors+1:end);
%extractors_normed = normify(extractors);

% subplot (3,3,8);
% if simparams.doPassValidationTrialsWithTraining
%     rpidxs = randperm(length(simparams.validationTrialIdxs));
%     vplot_idx = simparams.validationTrialIdxs(rpidxs(1));
%     imagesc(time, [], r_output_T{vplot_idx});
% else
%     f_extracted_t = extractors'* m_targettrain_T{true_plot_idx};
%     thing1 = min(f_extracted_t');
%     thing2 = max(f_extracted_t');
%     offset = max(abs(thing2 - thing1));
%     for i = 1:nfactors
%         plot(time, f_extracted_t(i,:)+(i-1)*offset, 'c', 'linewidth', 2);
%         hold on;
%     end
%     hold off;
%     axis tight;
% end
%% Visualize recurrent features.
subplot (3,3,9);
if net.dt == net.tau
    x = linspace(-1,1,100);
    y = sqrt(1-x.^2);
    plot(x, y, 'k');
    hold on; 
    plot(x, -y, 'k');
else
    y = linspace(-1,1,100);
    x = 0;
    plot(x, y, 'k');
    hold on;
end
plot(eig(rnet_mat), 'rx', 'linewidth', 2);
hold off;


