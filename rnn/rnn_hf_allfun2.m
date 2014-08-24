function varargout = rnn_hf_allfun2(net, v_u_t, m_target_t, wc, v, lambda, forward_pass, ...
    training_vs_validation, trial_id, optional_params, simdata, ...
    do_return_network, do_return_L, do_return_L_grad, ...
    do_return_L_GaussNewton, do_return_preconditioner)
% Written by David Sussillo (C) 2013
%
%function varargout = rnn_hf_allfun_trace2(net, v_u_t, m_target_t, wc, v, lambda, forward_pass, ...
%    training_vs_validation, trial_id, optional_params, ...
%    do_return_network, do_return_L, do_return_L_grad, ...
%    do_return_L_GaussNewton, do_return_preconditioner)
%
%
% This function will do it all, that's the only way I know how to keep the code from having multiple implementations.
%
%
% A layer is respect to the weights, so input -> weights -> recurrent  ( layer 1 )
%                                       recurrent -> weights -> recurrent  ( layer 2 )
%                                       recurrent -> weights -> output  ( layer 3 )
%
% net - the network structure.
% v_u_t - the input to the network.
%
% m_target_t - the targets for the network
% (can be NaN, or whatever, as it's ignored.)
%
% wc - weight cost
% v - The vector used to find the Hessian vector product Hv in Hessian-free
% learning.
%
% lambda - the regularization parameter for HF learning.
%
% forward_pass - the network activity cell array as an optimization so we
% don't have recompute forward passes or their derivatives over and over
% again.
%
% training_vs_validation is 1 for training and 2 for validation
%
% trial_id - The trial id is with respect to either the training set or the validation
% set.
%
% optional_params - a completely user determined structure passed in from
% the hfopt.m routine for whatever purposes the network designer wishes.
% So far, in this case I used it for getting condition dependent initial conditions.
%
% do_return_* - What parts of the computation should this pass through the
% routine compute?


% WARNING 07/2014.  I don't think that the jacobian regularizer and the L2 regularizer should be active at the same time, except for the output weights for L2.
% The reason I write this is because now that the Jreg back-propagates, all the weights, including inputs, biases, initial conditions, etc, are
% subject to its power to make a smoother system.

%% Basic setup

% in hfopt2.m, eval is inputs -> forward_pass -> targets.  This means that the targets may be empty for the forward pass.
if isempty(m_target_t)
    assert ( do_return_network && ~(do_return_L || do_return_L_grad || do_return_L_GaussNewton || do_return_preconditioner ), 'Stopped');  
end   
if ~isempty(m_target_t)
    assert ( size(v_u_t,2) == size(m_target_t,2), 'Inputs and targets time length should be the same!');
end
    
if ( do_return_preconditioner )
    assert ( false, 'Preconditioner not supported for RNNs.'); % plus, what happens to averaging across trials?
end

% Should we learn the biases?
do_learn_biases = true;
if isfield(net, 'doBiases')
    do_learn_biases = net.doBiases;
end
if isfield(net, 'doLearnBiases');
    do_learn_biases = net.doLearnBiases;
end

% Should we learn the state initialization, x(0)?
do_learn_state_init = true;
ic_T_add = 0;
if isfield(net, 'doLearnStateInit')
    do_learn_state_init = net.doLearnStateInit;
    ic_T_add = 1;
end

[n_Wru_v, n_Wrr_n, m_Wzr_n, n_ICx0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);


[V,T] = size(v_u_t);		% get relevant dimensions
[M,N] = size(m_Wzr_n);


% Transfer functions
rec_trans_fun = net.layers(2).transFun;
out_trans_fun = net.layers(3).transFun;
rec_deriv_fun = net.layers(2).derivFunAct;
rec_deriv2_fun = net.layers(2).deriv2FunAct;
out_deriv_fun = net.layers(3).derivFunAct;

do_one_pool = true;
if isfield(net, 'nPools')
    npools = net.nPools;
    pool_idxs = net.poolIdxs;
    do_one_pool = false;
    rec_trans_funs = net.recTransFuns;
    rec_deriv_funs = net.recDerivFuns;
    rec_deriv2_funs = net.recDeriv2Funs;
end
if do_one_pool
    npools = 1;
    p = 1;
    pool_idxs{p} = 1:N;
    rec_trans_funs = cell(1,npools);
    rec_deriv_funs = cell(1,npools);
    rec_deriv2_funs = cell(1,npools);
    for p = 1:npools
        rec_trans_funs{p} = rec_trans_fun;
        rec_deriv_funs{p} = rec_deriv_fun;
        rec_deriv2_funs{p} = rec_deriv2_fun;
    end
end


% Get the noise value.
noise_sigma = 0.0;
if ( isfield(net, 'noiseSigma') )
    noise_sigma = net.noiseSigma;
end

% DT / TAU, which figures into the Euler integration.  If DT / TAU = 1,
% it's a discrete network, so we've folded both cases in here.
dt_o_tau = net.dt / net.tau;
assert ( dt_o_tau <= 1.0, 'DT/TAU cannot be greater than 1.0');

%% Initial conditions

% Check for CONDITION DEPENDENT, NON-LEARNED ICS.
% THIS IS FOR THE CASE WHEN YOU DON'T LEARN THE ICS.
% There are a ton of applications of RNNs that require already being in
% the right place in state space. On the other hand, at the end of a trial, we have exactly
% the network state in the right place.  So let's save it, in some
% meaningfully indexed way for use later.
% If you want to LEARN the ICs, instead of merely SAVING them, you have to use the net.theta parameter
% array, because learning implies Hv type stuff.
%
% This is to load the beginning state. It's the first IC in the first run of the system. Note *_t variables.
% Note also that the save portion of this is accomplished via a user hook
% function in hfopt.m, optional_network_update_fun, which allows the forward
% pass to update the network structure.  It can't happen here because this
% function is inside a some kind of parallel context.
do_use_saved_ics = false;
if isfield(net, 'savedICs') && net.savedICs.doUseSavedICs   % short-circuit
    do_use_saved_ics = true;
    get_or_save = 1;
    gic_idx = net.savedICs.getICIndex(net, v_u_t, m_target_t, training_vs_validation, trial_id, get_or_save);
    n_sx0_1 = net.savedICs.ICs(:,gic_idx);
end

% This is TRIAL DEPENDENT, UNLEARNABLE ICS.  One could imagine this as a
% noise parameter for all trials, including validation trials.
do_use_trial_ics = false;
if isfield(net, 'trialICs') && net.trialICs.doUseTrialICs
    do_use_trial_ics = true;
    n_tx0_1 = net.trialICs.ICs{training_vs_validation}(:,trial_id);  % This matrix is N x ntrials
%    norm(n_tx0_1)
end

% This is trial_dependent noise on the initial condition.  This is useful if you want to learn the initial condition, but you also want to have a
% ball of noise around it.  I'm using the more modern simdata to implement this.  It's gendata's responsibility to generate the noise.  I add it
% here.  This should work seemlessly with multiple ICs (multiple conditions), as this will simply be appended to the IC that is choosen.
do_use_trial_dependent_ic_noise = false;
if isfield(simdata, 'doTrialDependentICNoise') && simdata.doTrialDependentICNoise
    do_use_trial_dependent_ic_noise = true;
    n_x0_ic_noise_1 = simdata.x0ICNoise;
end


% LEARNABLE, CONDITION DEPENDENT INITIAL CONDITIONS.  So
% far, it's not been that helpful.
condition_id = 1;
n_addinput_t = [];
noise_seed = 0;
if ( ~isempty(optional_params) ) % Optional params is struct that has things in it, already just for this one case.
    assert ( isstruct(optional_params), 'No longer a cell array.' );
    if ( isfield(optional_params, 'conditionID') )
        condition_id = optional_params.conditionID;
    end
    if ( isfield(optional_params, 'additionalInput') )
        n_addinput_t = optional_params.additionalInput;  % This had better be NxT, or I'ma break that face!
    end
    if ( isfield(optional_params, 'noiseSeed') )
        noise_seed = optional_params.noiseSeed;
    end
    % ...
    
end
n_bx0_1 = n_ICx0_c(:,condition_id);


% There are multiple IC mechanisms availiable.  Let's make sure only one is active. 
assert ( sum([do_use_trial_dependent_ic_noise do_use_trial_ics do_use_saved_ics]) <= 1, 'stopped');  % Zero is fine.
if do_use_saved_ics
    assert ( do_use_trial_ics == false, 'ICs are wrong.');
    n_x0_1 = n_sx0_1;
elseif do_use_trial_ics
    assert ( do_use_saved_ics == false, 'ICs are wrong.');
    n_x0_1 = n_tx0_1;
elseif do_use_trial_dependent_ic_noise
    n_x0_1 = n_bx0_1 + n_x0_ic_noise_1; 
else
    n_x0_1 = n_bx0_1;
end

n_r0_1 = zeros(N,1);
n_dr0_1 = zeros(N,1);
for p = 1:npools
    pidxs = pool_idxs{p};
    n_r0_1(pidxs) = rec_trans_funs{p}(n_x0_1(pidxs));
    n_dr0_1(pidxs) = rec_deriv_funs{p}(n_r0_1(pidxs));
end


%% Value masks.
% The mask for times that have no nans at all.
%tmask = ~isnan(sum(m_target_t,1));
vmask = ~isnan(m_target_t);     % Use logical indexing to allow a single time index with both a value and NaN. DCS:2/15/2012
ntargets = length(find(vmask));
%assert ( ntargets > 0, 'Something wrong here.');
assert ( M > 0, 'Something wrong here.');
% Allow all nans, so we short-ciruit some computation below.
do_bother_computing_L = sum(vmask(:)) > 0;

TxM_vmasked = ntargets;  % Note that eval_trials.m is handling the normalization by # samples (RNN runs).


%% Modification mask

% We allow 'parameters' that aren't modified.  It's easier to represent
% blocks of parameters than to get with a sparse representation.
mod_mask = ones(size(net.costMask));
if isfield(net, 'modMask')
    mod_mask = net.modMask;
end

cost_mask = net.costMask;
npieces = 6;  % number of things unpacked by unpackRNN
mod_mask_by_trial = ones(1,npieces);
if isfield(net, 'modMasksByTrial')
    mod_mask_by_trial = net.modMasksByTrial(:,condition_id);
end
WruMM_trial = mod_mask_by_trial(1);
WrrMM_trial = mod_mask_by_trial(2);
WzrMM_trial = mod_mask_by_trial(3);
x0MM_trial = mod_mask_by_trial(4);
bxMM_trial = mod_mask_by_trial(5);
bzMM_trial = mod_mask_by_trial(6);


%% Regularizers, etc.

do_recrec_Frobenius_norm_regularizer = false;
if isfield(net, 'frobeniusNormRecRecRegularizer')
    if net.frobeniusNormRecRecRegularizer > 0.0 
        do_recrec_Frobenius_norm_regularizer = true;
        froRR = net.frobeniusNormRecRecRegularizer;  % Cost on Frobenius norm.
    end
end

do_firing_rate_mean_regularizer = false;
if isfield(net, 'lowFiringRateRegularizer')
    assert ( false, 'Need to change parameterizertion for low firing rate regularizer.');
end

if isfield(net, 'firingRateMean')
    if net.firingRateMean.weight > 0.0
        do_firing_rate_mean_regularizer = true;
        
        fr_mean_reg_weight = net.firingRateMean.weight;
        fr_mean_reg_dv = net.firingRateMean.desiredValue;
        fr_mean_reg_mask = logical(net.firingRateMean.mask);       
        
        N_fr_mean_reg = length(find(fr_mean_reg_mask));
        
        assert ( size(fr_mean_reg_mask,1) == N, 'stopped');
        assert ( N_fr_mean_reg > 0, 'stopped' );
    end
end

do_firing_rate_var_regularizer = false;
if isfield(net, 'firingRateVariance')
   if net.firingRateVariance.weight > 0.0
       do_firing_rate_var_regularizer = true;
       
       fr_var_reg_weight = net.firingRateVariance.weight;
       fr_var_reg_dv = net.firingRateVariance.desiredValue;
       fr_var_reg_mask = logical(net.firingRateVariance.mask);
       
       N_fr_var_reg = length(find(fr_var_reg_mask));
       
       assert ( size(fr_var_reg_mask,1) == N, 'stopped');
       assert ( N_fr_var_reg > 0, 'stopped' );       
   end
end

do_firing_rate_covar_regularizer = false;
if isfield(net, 'firingRateCovariance')
   if net.firingRateCovariance.weight > 0.0
       do_firing_rate_covar_regularizer = true;
       
       fr_covar_reg_weight = net.firingRateCovariance.weight;
       fr_covar_reg_dv = net.firingRateCovariance.desiredValue;
       fr_covar_reg_mask = logical(net.firingRateCovariance.mask);
       
       N_fr_covar_reg = length(find(fr_covar_reg_mask));
       
       assert ( size(fr_covar_reg_mask,1) == N, 'stopped');       
       assert ( N_fr_covar_reg > 0, 'stopped' );       
   end
end


do_norm_pres_regularizer = false;
if isfield(net, 'normedPreWeights')
    if net.normedPreWeights.weight > 0.0
        do_norm_pres_regularizer = true;
        
        norm_pres_reg_weight = net.normedPreWeights.weight;
        norm_pres_reg_dv = net.normedPreWeights.desiredValue;
        norm_pres_reg_mask = logical(net.normedPreWeights.mask);       
        
        N_norm_pres_reg = length(find(norm_pres_reg_mask));
        
        assert ( size(norm_pres_reg_mask,1) == N, 'stopped');
        assert ( N_norm_pres_reg > 0, 'stopped' );
    end
end

is_training = false;
if isfield(net, 'isTraining')
    is_training = net.isTraining;
end

%% Forward pass
if ( isempty(forward_pass) )
    
    n_x_1 = n_x0_1;
    n_r_1 = n_r0_1;
    
    % Note that if call to a random number generator happens between here and the randn call for the noise, this setup will break.
    if noise_sigma > 0.0 && noise_seed > 0
        rng(noise_seed);
    end
    
    % Forward pass up to nonlinear network outputs, but not loss.
    n_x_t = zeros(N,T);
    n_r_t = zeros(N,T);
    n_Wu_t = n_Wru_v * v_u_t;	% A little parallelism
    n_nnoise_t = zeros(N,T);
    if ( noise_sigma > 0.0 )
        n_nnoise_t = noise_sigma * randn(N,T);
    end
    if ( ~isempty(n_addinput_t) )
        n_nnoise_t = n_nnoise_t + n_addinput_t;
    end
    for t = 1:T
        n_x_1 = (1.0-dt_o_tau)*n_x_1 + dt_o_tau*( n_Wu_t(:,t) + n_Wrr_n*n_r_1 + n_bx_1 + n_nnoise_t(:,t) );
        %n_r_1 = rec_trans_fun( n_x_1 );
        for p = 1:npools
            pidxs = pool_idxs{p};
            n_r_1(pidxs) = rec_trans_funs{p}(n_x_1(pidxs));
        end

        n_x_t(:,t) = n_x_1;
        n_r_t(:,t) = n_r_1;
    end
    
    n_dr_t = zeros(N,T);
    for p = 1:npools
        pidxs = pool_idxs{p};
        n_dr_t(pidxs,:) = rec_deriv_funs{p}(n_r_t(pidxs,:));
    end

    m_z_t = out_trans_fun( m_Wzr_n * n_r_t + repmat(m_bz_1, 1, T) ); % a little parallelism
    
else
    n_r_t = forward_pass{1};		
    n_dr_t = zeros(N,T);
    for p = 1:npools
        pidxs = pool_idxs{p};
        n_dr_t(pidxs,:) = rec_deriv_funs{p}(n_r_t(pidxs,:));
    end
    m_z_t = out_trans_fun( m_Wzr_n * n_r_t + repmat(m_bz_1, 1, T) ); % a little parallelism
end


%% Compute the Objective Function
if ( do_return_L )
    
    % We allow for NaN in target to signify no value, so we have to check for it.
    %m_target_tv = m_target_t(:, tmask); % need not be contiguous in time.
    %m_z_tv = m_z_t(:,tmask);	% ""
    all_Ls = [];
    
    if do_bother_computing_L
        mtv_target_1 = m_target_t(vmask); % need not be contiguous in time.
        mtv_z_1 = m_z_t(vmask);	% ""
        
        switch net.objectiveFunction
            case 'cross-entropy'
                switch net.layers(end).type
                    case 'cross-entropy'
                    case 'logistic'
                        L_output = -sum(sum(mtv_target_1 .* log(mtv_z_1+realmin) + (1 - mtv_target_1).*log(1-mtv_z_1+realmin)));
                    case 'softmax'
                        L_output = -sum(sum(mtv_target_1 .* log(mtv_z_1+realmin)));
                    otherwise
                        disp('Eat shit and die.');
                end
            case 'sum-of-squares'
                L_output = (1.0/2.0) * sum( sum( (mtv_target_1 - mtv_z_1).^2 ));
            case 'nll-poisson'
                % warning('This log(K!) is numerically unstable if K! gets large.');
                % The right way to do this is to have \sum^K log(k), for log(K!).  But it could be slow, and since
                % our first application won't have more than 3 or 4 spikes (max) per time bin, I'm ignoring for now.
                L_output = -sum(sum( mtv_target_1 .* log(mtv_z_1+realmin) - mtv_z_1 - log(factorial(mtv_target_1)+realmin) ));
            otherwise
                disp('Objective function not implemented.');
        end
        L_output = L_output/TxM_vmasked;
        all_Ls(end+1) = L_output; %#ok<*NASGU>
        
        % Even though I'm unsure about dividing by ntargets below, it seems like this is a no-brainer.  It scales the
        % objective function by the number of meaningful targets, making the objective function more easily interprettable,
        % and I think also allowing for weight_cost parameter to scale effectively. Keep in mind that rho uses this value
        % to get the denominator, so if you normalize here you have to for the grad and hessian as well, even if it does
        % cancel out there in the Newton step. -DCS:2011/09/14
        
        % Weight mask determines which weights have a cost.  Only weights that
        % are being modifed are subject to cost.
        % Note that there is nothing trial specific about this.  So it's computed over and over, and then divided by the number
        % of times it's computed.
        L_l2weight = (wc/2.0)*sum((mod_mask .* cost_mask .* net.theta).^2);
        all_Ls(end+1) = L_l2weight;
        
        L_fro = 0;
        if do_recrec_Frobenius_norm_regularizer      
            if isfield(net, 'frobNormRowIdxs')  %  A reasonable hack for now, allows you to choose a submat to regularize.
                frob_row_idxs = net.frobNormRowIdxs;
                frob_col_idxs = net.frobNormColIdxs;
            else
                frob_row_idxs = 1:N;
                frob_col_idxs = 1:N;
            end
            
            fr_WrrFN_fc = n_Wrr_n(frob_row_idxs,frob_col_idxs);
            fc_drFN_tpl = [n_dr0_1(frob_col_idxs) n_dr_t(frob_col_idxs,:)];
            
            % froRR / (2T) sum_j sum_t r'(t)^2_j sum_i Jij^2
            frob_factor = froRR/(2.0*(T+ic_T_add));
            sW2_fc = sum(fr_WrrFN_fc.^2,1);   % sum_i W_ij^2
            sdr2_fc = sum(fc_drFN_tpl.^2, 2)'; % sum_t (r'(t)_j)^2
            s_sW2_sdr2 = sum(sW2_fc .* sdr2_fc);   % sum_j sum_t (r'(t)_j)^2 sum_i W_ij^2
            L_fro = frob_factor * s_sW2_sdr2;     % Average over T to get the squared Frobenius norm.
        end
        all_Ls(end+1) = L_fro;
        
        % Low firing rate regularization
        % Note that these WILL still be in play even if T is masked.
        L_fr_avg = 0;
        if do_firing_rate_mean_regularizer        
            % This is L = \lambda / (2 * N) \sum_i^N ( 1/T * \int_0^T r_i(t') dt' - \alpha)^2.            
            n_r_avg_1 = mean(n_r_t,2);
            n_error_in_avg_1 = fr_mean_reg_mask .* (n_r_avg_1 - fr_mean_reg_dv);
            L_fr_avg = (fr_mean_reg_weight/(2.0*N_fr_mean_reg)) * sum( n_error_in_avg_1.^2 );  % So this will penalize the average firing rate over time.
        end     
        all_Ls(end+1) = L_fr_avg;
        
        L_fr_var = 0;
        if do_firing_rate_var_regularizer
            % This is L = \lambda / (2 * N) \sum_i^N ( 1/T \int_0^T (r_i(t') - <r_i>)^2 dt' - \alpha )^2            
            n_r_avg_1 = (1/T)*sum(n_r_t, 2);
            n_r_var_1 = (1/T)*sum((n_r_t - repmat(n_r_avg_1, 1, T)).^2, 2);
            n_error_in_var_1 = fr_var_reg_mask .* (n_r_var_1 - fr_var_reg_dv);
            L_fr_var = (fr_var_reg_weight/(2.0*N_fr_var_reg)) * sum( n_error_in_var_1.^2 );                                                                                       
        end        
        all_Ls(end+1) = L_fr_var;
        
        L_fr_covar = 0;
        if do_firing_rate_covar_regularizer
            n_r_avg_1 = (1/T)*sum(n_r_t, 2);
            n_rma_t = repmat(fr_covar_reg_mask, 1, T) .* (n_r_t - repmat(n_r_avg_1, 1, T));
            n_rcov_n = (1/T) * (n_rma_t * n_rma_t');
            n_dvI_n = (eye(N,N) * fr_covar_reg_dv) .* repmat(fr_covar_reg_mask, 1, N);
            L_fr_covar = (fr_covar_reg_weight / (2.0 * N_fr_covar_reg^2)) * sum ( sum ( (n_rcov_n - n_dvI_n^2).^2 ) );
        end
        all_Ls(end+1) = L_fr_covar;
        
        L_normed_pre_weights = 0;
        if do_norm_pres_regularizer                        
            % Normify works on columns, but we're norming inputs, so we work on rows.            
            % Get away without the square root by squaring both the norm and the desired value.
            [~, l_prenorms_m] = normify(n_Wrr_n(norm_pres_reg_mask,:)');  
            m_prenorms2_1 = (l_prenorms_m').^2;
            L_normed_pre_weights = (norm_pres_reg_weight /(2.0 * N_norm_pres_reg)) * sum (m_prenorms2_1 - norm_pres_reg_dv^2 ).^2;
        end
        all_Ls(end+1) = L_normed_pre_weights;
    end
    L = sum(all_Ls);
end

%% Backprop through time for gradient of RNN weights.
if ( do_return_L_grad )
    
    if ( dt_o_tau < 1.0 )
        n_Wrrt_dec_n = dt_o_tau * n_Wrr_n';	 % transpose this mofo, and decay term for continuous system
        n_xdec_n = (1.0-dt_o_tau);	         % as if it were identity times scalar
    else
        n_Wrrt_dec_n = n_Wrr_n';
        n_xdec_n = 0;		                 % as if it were zero matrix
    end
    n_Wrzt_m = m_Wzr_n';	                 % transpose this mofo
    
    m_dLdy_t = zeros(M,T);
    if ( net.hasCanonicalLink )
        m_dLdy_t(vmask) = m_z_t(vmask) - m_target_t(vmask);
    else
        assert ( false, 'Double check this for the specific case.');
        m_dLdy_t(vmask) = out_deriv_fun(m_z_t(vmask)) .* (m_z_t(vmask) - m_target_t(vmask));
    end
    m_dLdy_t = m_dLdy_t / TxM_vmasked;
    
    % Regularizers and firing rate controls.
    % Note that I'm explicitly not back-propagating the extras to r0, since firing rates, etc. may not make sense for
    % the IC.
    n_dLextrasr_tp1 = zeros(N,T+1);
    if do_recrec_Frobenius_norm_regularizer  % This the back-propagated part of the Frobenius norm derivative.  There's also a direct part (below).
        frob_factor = froRR/(2.0*(T+ic_T_add));
        
        n_d2r_t = zeros(N,T);
        n_d2r0_1 = zeros(N,1);
        for p = 1:npools
            pidxs = pool_idxs{p};
            n_d2r_t(pidxs,:) = rec_deriv2_funs{p}(n_r_t(pidxs,:), n_dr_t(pidxs,:));
            n_d2r0_1(pidxs) = rec_deriv2_funs{p}(n_r0_1(pidxs), n_dr0_1(pidxs));
        end
        
        if isfield(net, 'frobNormRowIdxs')  %  A reasonable hack for now, allows you to choose a submat to regularize.
            frob_row_idxs = net.frobNormRowIdxs;
            frob_col_idxs = net.frobNormColIdxs;
        else
            frob_row_idxs = 1:N;
            frob_col_idxs = 1:N;
        end
        
        fr_WrrFN_fc = n_Wrr_n(frob_row_idxs, frob_col_idxs);
        fc_drFN_tpl = [n_dr0_1(frob_col_idxs) n_dr_t(frob_col_idxs,:)];
        fc_d2rFN_tp1 = [n_d2r0_1(frob_col_idxs) n_d2r_t(frob_col_idxs,:)];
        
        % This is \sum_t^T \del L/\del x_j(t) * \del x_j(t)\del_{J_kl}
        % \del L / \del x_j(t) = (fro/(2*T)) * (\sum_i W_ij^2) 2 r'_j(t) r''_j(t)
        %          
        % This means we should back-propagate \del L / \del x_j(t)
        sW2_fc = sum(fr_WrrFN_fc.^2,1);   % sum_i W_ij^2
        fc_dLfdxFN_tp1 = frob_factor * bsxfun(@times, 2.0 * fc_drFN_tpl .* fc_d2rFN_tp1, sW2_fc');   % sum_j sum_i W_ij^2 sum_t 2 r_j'(t) r_j''(t) 
        n_dLfdx_tp1 = zeros(N,T+1);
        n_dLfdx_tp1(frob_col_idxs, :) = fc_dLfdxFN_tp1;
        n_dLextrasr_tp1 = n_dLextrasr_tp1 + n_dLfdx_tp1;
    end
    if do_firing_rate_mean_regularizer
        ravg_factor = fr_mean_reg_weight / (N_fr_mean_reg * T);
        n_dLFRdr_avg_1 = ravg_factor * ((mean(n_r_t,2) - fr_mean_reg_dv) .* fr_mean_reg_mask);
        n_dLextrasr_tp1(:,2:T+1) = n_dLextrasr_tp1(:,2:T+1) + repmat(n_dLFRdr_avg_1, 1, T);
    end    
    if do_firing_rate_var_regularizer
        rvar_factor = fr_var_reg_weight / (N_fr_var_reg) * 2 * (T-1)/(T^2);
        n_r_avg_1 = (1/T)*sum(n_r_t, 2);
        n_rma_t = n_r_t - repmat(n_r_avg_1, 1, T);
        n_r_var_1 = (1/T)*sum(n_rma_t.^2, 2);
        n_dLFRdr_var_t = rvar_factor * (repmat((n_r_var_1 - fr_var_reg_dv) .* fr_var_reg_mask, 1, T) .* n_rma_t);
        n_dLextrasr_tp1(:,2:T+1) = n_dLextrasr_tp1(:,2:T+1) + n_dLFRdr_var_t;
    end
    if do_firing_rate_covar_regularizer
        rcovar_factor = fr_covar_reg_weight / (N_fr_covar_reg^2) * 2 * (T-1)/(T^2);
        n_dvI_n = (eye(N,N) * fr_covar_reg_dv) .* repmat(fr_covar_reg_mask, 1, N);
        n_r_avg_1 = (1/T)*sum(n_r_t, 2);
        n_rma_t = (n_r_t - repmat(n_r_avg_1, 1, T)) .* repmat(fr_covar_reg_mask, 1, T);
        n_rcov_n = (1/T) * (n_rma_t * n_rma_t');
        n_dLFRdr_covar_t = rcovar_factor * ((n_rcov_n - n_dvI_n) * n_rma_t);
        n_dLextrasr_tp1(:,2:T+1) = n_dLextrasr_tp1(:,2:T+1) + n_dLFRdr_covar_t;
    end
    
    % Backward pass.
    n_dLdx_t = zeros(N,T);
    n_dLdx_1 = zeros(N,1);
    for t = T:-1:1			% backward pass
        m_dLdy_1 = m_dLdy_t(:,t);
        n_dLdx_1 = n_xdec_n * n_dLdx_1 + n_dr_t(:,t) .* (n_Wrrt_dec_n * n_dLdx_1 + n_Wrzt_m * m_dLdy_1 + n_dLextrasr_tp1(:,t+1));
        n_dLdx_t(:,t) = n_dLdx_1;
    end
    n_dLdx0_1 = zeros(N,1);
    if ( do_learn_state_init )	        
        % Verified that the IC training works for the LFR reg with multiple conditions.  The idea is that if you want your average firing rate to be
        % R0, then why wouldn't n_r0_1 to be part of this average as well? Not measured in error, though.
        n_dLdx0_1 = n_xdec_n * n_dLdx_1 + n_dr0_1 .* (n_Wrrt_dec_n * n_dLdx_1) + n_dLextrasr_tp1(:,1); % all the way to zero!
    end
    
    % Now update the the derivatives wrt to the weights. The direct application of the chain rule for partial derivatives does
    % not contain a mean averaging, it's a sum, so there's no mean normalization here. The chain rule is used to get
    % dLdx(t) (that's not normalized by T above) and also to get dLdtheta from dLdx_i * dxi/dtheta
    [n_WruMM_v, n_WrrMM_n, m_WzrMM_n, n_x0MM_c, n_bxMM_1, m_bzMM_1] = unpackRNNUtils(net, 'domodmask', true);
    t_rt_n = n_r_t';			% transpose for speed
    % m_dLdy_t is already vmasked
    n_rm1_t = [n_r0_1 n_r_t(:,1:end-1)];
    m_dLdWzr_n = WzrMM_trial * (m_WzrMM_n .* (m_dLdy_t * t_rt_n));% a little parallelism
    n_dLdWru_v = WruMM_trial * (n_WruMM_v .* (n_dLdx_t * v_u_t'));	           % a little parallelism
    n_dLdWrr_n = WrrMM_trial * (n_WrrMM_n .* (n_dLdx_t * n_rm1_t'));% a little parallelism
    
    if ( do_learn_biases )
        n_dLdbx_1 = bxMM_trial * (n_bxMM_1 .* sum(n_dLdx_t, 2));
        % m_dLdy_t is already vmasked
        m_dLdbz_1 = bzMM_trial * (m_bzMM_1 .* sum(m_dLdy_t, 2));	           % a little parallelism
    else
        n_dLdbx_1 = zeros(N,1);
        m_dLdbz_1 = zeros(M,1);
    end
    
    if ( dt_o_tau < 1.0 )		% Multiply the dt/\tau factor that comes from dx/dW (e.g. r_j(t-1))
        n_dLdWru_v = dt_o_tau * n_dLdWru_v;
        n_dLdWrr_n = dt_o_tau * n_dLdWrr_n;
        n_dLdbx_1 = dt_o_tau * n_dLdbx_1;
    end
    
    % Pack it up, pack it in.
    n_dLdx0_c = zeros(N,net.nICs);
    n_dLdx0_c(:,condition_id) = x0MM_trial * (n_x0MM_c(:,condition_id) .* n_dLdx0_1);
    grad = packRNN(net, n_dLdWru_v, n_dLdWrr_n, m_dLdWzr_n, n_dLdx0_c, n_dLdbx_1, m_dLdbz_1);
    
    % Add the weight decay terms.
    grad = grad + wc * (mod_mask .* cost_mask .* net.theta);
    
    % The non-backpropped part of the frobenius norm error gradient.
    % The backpropped part was handled above.
    if do_recrec_Frobenius_norm_regularizer                
        if isfield(net, 'frobNormRowIdxs')  %  A reasonable hack for now, allows you to choose a submat to regularize.
            frob_row_idxs = net.frobNormRowIdxs;
            frob_col_idxs = net.frobNormColIdxs;
        else
            frob_row_idxs = 1:N;
            frob_col_idxs = 1:N;
        end
        
        Nfr = length(frob_row_idxs);
        fr_WrrFN_fc = n_Wrr_n(frob_row_idxs, frob_col_idxs);
        fc_drFN_tpl = [n_dr0_1(frob_col_idxs) n_dr_t(frob_col_idxs,:)];  % By including n_dr0_1, we're regularizing the smoothness of the ICs.
        fc_dr2FN_tp1 = fc_drFN_tpl.^2;
        
        frob_factor = froRR/(2.0*(T+ic_T_add));
        sdr2_fc = sum(fc_dr2FN_tp1, 2)'; % sum_t r'_j(t)^2
        fr_grad_fro_FN_fc = frob_factor * (2.0 * fr_WrrFN_fc .* repmat(sdr2_fc, Nfr, 1));  %J_ij \sum_t r'_j(t)^2
        n_grad_fro_n = zeros(N,N);
        n_grad_fro_n(frob_row_idxs,frob_col_idxs) = fr_grad_fro_FN_fc;
        n_grad_fro_n = WrrMM_trial * n_WrrMM_n .* n_grad_fro_n;

        % Package it.
        grad_fro = packRNN(net, zeros(N,V), n_grad_fro_n, zeros(M,N), zeros(N,net.nICs), zeros(N,1), zeros(M,1));
        grad = grad + grad_fro;
    end   
    
    if do_norm_pres_regularizer
        % Normify works on columns, but we're norming inputs, so we work on rows.
        m_Wrr_n = n_Wrr_n(norm_pres_reg_mask,:);
        [~, l_prenorms_m] = normify(m_Wrr_n');
        m_prenorms2_1 = (l_prenorms_m.^2)';
        m_prenorm2_res_1 = (m_prenorms2_1 - norm_pres_reg_dv^2);
        
        fac = 2.0 * norm_pres_reg_weight/N_norm_pres_reg;
        
        n_grad_norm_pres_n = zeros(N,N);
        n_grad_norm_pres_n(norm_pres_reg_mask,:) = fac * (repmat(m_prenorm2_res_1, 1, N) .* m_Wrr_n);
        n_grad_norm_pres_n = WrrMM_trial * n_WrrMM_n .* n_grad_norm_pres_n;
        
        % Package it.
        grad_norm_pres = packRNN(net, zeros(N,V), n_grad_norm_pres_n, zeros(M,N), zeros(N,net.nICs), zeros(N,1), zeros(M,1));
        grad = grad + grad_norm_pres;                
    end
    
    % This finite difference checking routine is super important if you make fundamental changes to this routine.  Things get hairy REALLY quickly.
    do_check_grad = 0;
    if do_check_grad && norm(grad) > 0.01
        disp(['Norm of backprop gradient: ' num2str(norm(grad)) '.']);
        disp('Numerically checking the gradient created by backprop.');
   
        % Initialize numgrad with zeros
        theta = net.theta;
        numgrad = zeros(size(theta));
        EPS = 1e-4;        
        ngrads = size(theta(:),1);           
        eval_objfun = @(net) rnn_hf_allfun2(net, v_u_t, m_target_t, wc, [], [], [], 1, 1, optional_params, simdata, 0, 1, 0, 0, 0);
        
        for i = 1:ngrads
            e_i = zeros(ngrads,1);
            e_i(i) = 1;
            theta_i_minus = theta - EPS*e_i;
            theta_i_plus = theta + EPS*e_i;
            
            testnetp = net;
            testnetm = net;
            testnetp.theta = theta_i_plus;
            testnetm.theta = theta_i_minus;
            package = eval_objfun(testnetp);
            gradp = package{1};
            package = eval_objfun(testnetm);
            gradm = package{1};
            numgrad(i) = (gradp - gradm)/(2.0*EPS);
            
            if mod(i,1000) == 0
                disp(i);
            end
        end        
        
        [n_ngWru_v, n_ngWrr_n, m_ngWzr_n, n_ngx0_c, n_ngbx_1, m_ngbz_1] = unpackRNN(net, numgrad);
        [n_gWru_v, n_gWrr_n, m_gWzr_n, n_gx0_c, n_gbx_1, m_gbz_1] = unpackRNN(net, grad);
        
        % I believe these results are HIGHLY dependent on whether or not the gradient is exploding or vanishing.
        % If the gradient is exploding, this can look very, very ugly.  Makes one wonder how this works at all!
        % If the gradient is vanishing, then this looks very, very good: order 1e-10.  Otherwise, it's down to 1e-5 or worse.       
        disp(['n_Wru_v: ' num2str(mean(vec(abs(n_ngWru_v - n_gWru_v))))]);
        disp(['n_ICx_c: ' num2str(mean(vec(abs(n_ngx0_c - n_gx0_c))))]);
        disp(['n_Wrr_n: ' num2str(mean(vec(abs(n_ngWrr_n - n_gWrr_n))))]);
        disp(['n_bx_1: ' num2str(mean(vec(abs(n_ngbx_1 - n_gbx_1))))]);
        disp(['m_Wzr_n: ' num2str(mean(vec(abs(m_ngWzr_n - m_gWzr_n))))]);
        disp(['m_bz_1: ' num2str(mean(vec(abs(m_ngbz_1 - m_gbz_1))))]);
        
        diff = norm(numgrad-grad)/norm(numgrad+grad);
        disp(['norm(numgrad-grad)/norm(numgrad+grad): ' num2str(diff)]);
        fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');        
    end
end


%% Hv: Compute the GaussNewton forward and backward Pass
% Note the transposes here are important.
%
% F = L(M(N(w)))
%
% F:  weights -> loss
% N:  weights -> outputs (linear)
% M:  outputs (linear) -> outputs
% L:  outputs -> loss
%
% f0 is the ordinary forward pass of a neural network, evaulting the function F(w) it implements by propagating
% activity forward through F.
%
% r1 is the ordinary backward pass of a neural network, calculating J_F' u by propagating the vector u backward
% through F.  This pass uses intermediate results computed in the f0 pass.
%
% f1 is based on R_v(F(w)) = J_F v, for some vector v.  By pushing R_v, which obeys the usual rules for differential
% operators, down in the the equations of the forward pass f0, one obtains an efficient procedure to calculate J_F v.
%
% r2, when the R_v operator is applied to the r1 pass for a scalar function F, one obtains an efficient procedure for
% calculating the Hessian-vector product H_F v = Rv(J_F')
%
%
% This is from Schraudolph, Table 1.
% pass     f0    r1(u)      f1(v)   r2
% result   F    J_F' u      J_F v   H_F v

% The gradient g = J'_(L.M.N) is computed by an f0 pass through the entire model (N, M and L), followed by an r1 pass
% propagating u = 1 back through the entire model (L, M the N).  For macthing loss functions, there is a shortcut since
% J_(L.M) = Az+b (z-z*), we can limit the forward pass to N and M (to compute z) then r1-propagate u = Az+b (z-z*) back
% through just N.

% For matching loss functions, we do not require an r2 pass (for GN matrix).  Since
% G = J_N' H_(L.M) J_N = J_N' J_M' J_N,
% we can limit the f1 pass to N: weights -> outputs(linear)
% then r1 propagate it back through M and N.

% For linear with lsqr error,                     H_(L.M) = J_M = I
% For logistic function w/ cross-entroy loss,     H_(L.M) = diag(diag(z)(1-z))
% For softmax w/ cross-entropy loss,              H_(L.M) = diag(z) - zz'

% So get J_n v from f1-pass up to output (linear).
% Then multiply by H_(L.M)
% Once you set up H_(L.M) J_n v, then backprop gives you J_N' H_(L.M) J_N v
% J_N v is forward pass of R operation
if ( do_return_L_GaussNewton )   % GN start
    
    assert ( net.hasCanonicalLink, 'We require canonical link functions.' );
    mu = net.mu;
    
    % Any parameter that cannot be modified, is not a parameter.  Therefore R{w} = 0;  This is the equivalent of mod masking the derivatives in 
    % the gradient section.
    [n_VWru_v, n_VWrr_n, m_VWzr_n, n_vx0_c, n_vbx_1, m_vbz_1] = unpackRNN(net, v);
    [n_WruMM_v, n_WrrMM_n, m_WzrMM_n, n_x0MM_c, n_bxMM_1, m_bzMM_1] = unpackRNNUtils(net, 'domodmask', true);
    n_VWru_v = WruMM_trial * (n_WruMM_v .* n_VWru_v);
    n_VWrr_n = WrrMM_trial * (n_WrrMM_n .* n_VWrr_n);
    m_VWzr_n = WzrMM_trial * (m_WzrMM_n .* m_VWzr_n);
    %n_vx0_c = n_x0MM_c .* n_vx0_c;  % Handled differently below.
    n_vbx_1 = bxMM_trial * (n_bxMM_1 .* n_vbx_1);
    m_vbz_1 = bzMM_trial * (m_bzMM_1 .* m_vbz_1);
        
    % Leaving these in below, because I'm pretty sure that init_rnn.m doesn't do the right thing, namely, set the mod mask to 0 for these 
    % parameters below.  This should really be fixed.  DCS 1/28/2014
    n_vx0_1 = n_vx0_c(:,condition_id);  
    if do_learn_state_init  % If we learn the IC, then it's a parameter, so R{IC} = vIC.  Else, if the IC is a constant, R{IC} = 0
        n_Rx0_1 = x0MM_trial * (n_x0MM_c(:,condition_id).* n_vx0_1);
    else
        n_Rx0_1 = zeros(N,1);
    end
    if do_learn_biases  % If we learn the biases, then it's a parameter, so R{b^{x|z}}} = vb^{x|z}.  Else, it's a constant, so R{.} = 0.
        n_vbx_or_zero_1 = bxMM_trial * (n_bxMM_1 .* n_vbx_1);
        m_vbz_or_zero_1 = bzMM_trial * (m_bzMM_1 .* m_vbz_1);
    else
        n_vbx_or_zero_1 = zeros(N,1);
        m_vbz_or_zero_1 = zeros(M,1);        
    end    
    
    n_Rr0_1 = n_dr0_1 .* n_Rx0_1;
    n_Rx_1 = n_Rx0_1;
    n_Rr_1 = n_Rr0_1;
    
    % f1: Forward pass for R operation, so called f1 pass in Schraudolph, giving J_F (v)
    n_rm1_t = [n_r0_1 n_r_t(:,1:end-1)];
    n_VWruu_t = n_VWru_v * v_u_t;	% A little parallelism
    n_VWrrrm1_t = n_VWrr_n * n_rm1_t; 	% "" , note index offset here because IC is added to beginning
    m_VWzrr_t = m_VWzr_n * n_r_t;	% ""    
    n_Rr_t = zeros(N,T);		% not saving initialization bias    
    n_Rx_t = zeros(N,T);
    for t = 1:T 
        n_Rx_1 = (1.0-dt_o_tau) * n_Rx_1 + ...
            dt_o_tau * (n_vbx_or_zero_1 + n_VWruu_t(:,t) + n_VWrrrm1_t(:,1+t-1) + n_Wrr_n*n_Rr_1); % watch 1+t-1 cuz of bias
        n_Rr_1 = n_dr_t(:,t) .* n_Rx_1;  % becomes r'(t-1) * Rx(t-1) for next time step
        
        n_Rx_t(:,t) = n_Rx_1;
        n_Rr_t(:,t) = n_Rr_1;
    end
    
    % H_(L.M)
    m_Ry_t = repmat(m_vbz_or_zero_1,1,T) + m_VWzrr_t + m_Wzr_n*n_Rr_t;
    switch net.layers(end).type
        case 'softmax'
            m_Rz_t = m_Ry_t .* m_z_t  - m_z_t .* repmat( sum( m_Ry_t .* m_z_t, 1 ), [M 1] );
        otherwise
            m_dz_t = out_deriv_fun(m_z_t);
            m_Rz_t = m_dz_t .* m_Ry_t;    % H_(L.M) for m_Rz_t
    end
    
    % R backward pass, r1, and putting it altogether.  Should be exactly the backprop code above.
    if ( dt_o_tau < 1.0 )
        n_Wrrt_dec_n = dt_o_tau * n_Wrr_n';		% transpose this mofo, and continuous time constants
        n_xdec_n = (1.0-dt_o_tau);	                % as if it were identity matrix times a scalar
    else
        n_Wrrt_dec_n = n_Wrr_n';		% transpose this mofo
        n_xdec_n = 0.0;			        % as if it were a matrix of zeros
    end
    n_Wrzt_m = m_Wzr_n';		        % transpose this mofo     
    
    % backprop H_(L.M) J_N v to get Gv
    % J_N v is r1 pass (m_Rz_t)
    % H_(L.M) J_N v = A J_M J_N v is result of R forward pass with canonical link.
    m_RdLdy_t = zeros(M,T);
    if ( net.hasCanonicalLink )
        m_RdLdy_t(vmask) = m_Rz_t(vmask);
    else
        assert ( net.hasCanonicalLink, 'R{dLdy} not implemented for noncanonical link functions.' );
    end    
    
    m_RdLdy_t = m_RdLdy_t / TxM_vmasked;     
        
    % Firing rate controls.
    % These  are all GN approximations and not full Hessian.
    n_RdLextrasr_tp1 = zeros(N,T+1);
    if do_recrec_Frobenius_norm_regularizer
        frob_factor = froRR/(2.0*(T+ic_T_add));
        
        n_d2r_t = zeros(N,T);
        n_d2r0_1 = zeros(N,1);
        for p = 1:npools
            pidxs = pool_idxs{p};
            n_d2r_t(pidxs,:) = rec_deriv2_funs{p}(n_r_t(pidxs,:), n_dr_t(pidxs,:));
            n_d2r0_1(pidxs) = rec_deriv2_funs{p}(n_r0_1(pidxs), n_dr0_1(pidxs));
        end       
        
        if isfield(net, 'frobNormRowIdxs')  %  A reasonable hack for now, allows you to choose a submat to regularize.
            frob_row_idxs = net.frobNormRowIdxs;
            frob_col_idxs = net.frobNormColIdxs;
        else
            frob_row_idxs = 1:N;
            frob_col_idxs = 1:N;
        end
        
        fr_WrrFN_fc = n_Wrr_n(frob_row_idxs, frob_col_idxs);
        fc_d2rFN_tp1 = [n_d2r0_1(frob_col_idxs) n_d2r_t(frob_col_idxs,:)];
        fc_RxFN_tp1 = [n_Rx0_1(frob_col_idxs) n_Rx_t(frob_col_idxs,:)];
        
        sW2_fc = sum(fr_WrrFN_fc.^2,1);   % sum_i W_ij^2
        % WV_n = sum(n_Wrr_n .* n_VWrr_n); % Part of part1, which is not positive definite

        
        % These two parts are defined at the bottom of the page in Evernote, a note on the derivative 
        % and Hv products for the frobenius norm regularization.
        % part1alpha is implemented below.  It's a direct derivative and doesn't involve backpropagation.
        % part1beta = bsxfun(@times, n_dr_t .* n_d2r_t, (frob_factor * 2.0 * WV_n'));        % Not positive definite
        % part2alpha = ... % not implemented because not positive semi-definite
        fc_part2beta_FN_tp1 = bsxfun(@times, 2.0 * fc_d2rFN_tp1.^2 .* fc_RxFN_tp1, (frob_factor * sW2_fc'));
        % part2gamma = bsxfun(@times, 2.0*(n_dr_t .* n_d3r_t) .* n_Rx_t, (frob_factor * sW2_n'));  %(not positive semi-definite either cuz derivs could be negative)
        % part2delta = ... % not implemented because not postivie semi-definite
        %n_RdLextrasr_t = n_RdLextrasr_t + (part1 + part2);
        n_part2beta_tp1 = zeros(N,T+1);
        n_part2beta_tp1(frob_col_idxs,:) = fc_part2beta_FN_tp1;
        n_RdLextrasr_tp1 = n_RdLextrasr_tp1 + n_part2beta_tp1;
    end
    if do_firing_rate_mean_regularizer
        ravg_factor = fr_mean_reg_weight / (N_fr_mean_reg * T);
        n_RdLFRdravg_1 = ravg_factor * (mean(n_Rr_t,2) .* fr_mean_reg_mask);
        n_RdLextrasr_tp1(:,2:T+1) = n_RdLextrasr_tp1(:,2:T+1) + repmat(n_RdLFRdravg_1,1,T);
    end    
    if do_firing_rate_var_regularizer
        rvar_factor = fr_var_reg_weight / (N_fr_var_reg) * 2 * (T-1)/(T^2);
        n_r_avg_1 = (1/T)*sum(n_r_t, 2);
        n_rma_t = (n_r_t - repmat(n_r_avg_1, 1, T)) .* repmat(fr_var_reg_mask, 1, T);
        n_r_var_1 = (1/T)*sum(n_rma_t.^2, 2);

        n_Rr_avg_1 = (1/T)*sum(n_Rr_t, 2);
        n_Rrma_t = (n_Rr_t - repmat(n_Rr_avg_1, 1, T)) .* repmat(fr_var_reg_mask, 1, T);
        
        n_R_of_rvar_1 = (2/T) * sum((n_rma_t .* n_Rrma_t), 2);
        n_A_t = repmat(n_R_of_rvar_1, 1, T) .* n_rma_t;        
        n_B_t = repmat(n_r_var_1, 1, T) .* n_Rrma_t;
        
        n_RdLFRdr_var_t = rvar_factor * (n_A_t + n_B_t);
        n_RdLextrasr_tp1(:,2:T+1) = n_RdLextrasr_tp1(:,2:T+1) + n_RdLFRdr_var_t;
    end
    if do_firing_rate_covar_regularizer
        rcovar_factor = fr_covar_reg_weight / (N_fr_covar_reg^2) * 2 * (T-1)/(T^2);        
        n_r_avg_1 = (1/T)*sum(n_r_t, 2);        

        n_rma_t = (n_r_t - repmat(n_r_avg_1, 1, T)) .* repmat(fr_covar_reg_mask, 1, T);
        n_rcov_n = (1/T) * (n_rma_t * n_rma_t');
        
        n_Rr_avg_1 = (1/T)*sum(n_Rr_t, 2);
        n_Rrma_t = (n_Rr_t - repmat(n_Rr_avg_1, 1, T)) .* repmat(fr_covar_reg_mask, 1, T);
        
        n_Rrma_x_rma_n = (1/T)*(n_Rrma_t * n_rma_t');
        n_rma_x_Rrma_n = (1/T)*(n_rma_t * n_Rrma_t');
        n_Rrcov_n = n_Rrma_x_rma_n + n_rma_x_Rrma_n;
                
        n_RdLFRdr_covar_t = rcovar_factor * (n_Rrcov_n * n_rma_t + n_rcov_n * n_Rrma_t);
        n_RdLextrasr_tp1(:,2:T+1) = n_RdLextrasr_tp1(:,2:T+1) + n_RdLFRdr_covar_t;
    end 
    
    
    % Back-propagate R{y} to handle the GN wrt the output error.
    n_RdLdx_t = zeros(N,T);
    n_RdLdx_1 = zeros(N,1);
    lambda_mu = lambda * mu;
    for t = T:-1:1
        m_RdLdy_1 = m_RdLdy_t(:,t);
        n_RdLdx_1 = n_xdec_n * n_RdLdx_1 + n_dr_t(:,t) .* (n_Wrrt_dec_n * n_RdLdx_1 + n_Wrzt_m * m_RdLdy_1 + n_RdLextrasr_tp1(:,t+1));
        
        % Damping, do I have to to anything else to the error function?
        if ( mu > 0 )
            n_RdLdx_1 = n_RdLdx_1 + (lambda_mu * n_Rr_t(:,t));
        end
        n_RdLdx_t(:,t) = n_RdLdx_1;
    end
    
    n_RdLdx0_1 = zeros(N,1);
    if ( do_learn_state_init )	
        n_RdLdx0_1 = n_xdec_n * n_RdLdx_1 + n_dr0_1 .* ( n_Wrrt_dec_n * n_RdLdx_1) + n_RdLextrasr_tp1(:,1); % all the way to zero!
        if ( mu > 0 )
            n_RdLdx0_1 = n_RdLdx0_1 + (lambda_mu * n_dr0_1 .* n_Rx0_1 );
        end
    end    
    
    % Now update the R{} wrt to the weights. The direct application of the chain rule for partial derivatives does
    % not contain a mean averaging, it's a sum, so there's no mean normalization here. The chain rule is used to get
    % dLdx(t) (that's not normalized by T above) and also to get dLdtheta from dLdx_i * dxi/dtheta
    t_rt_n = n_r_t';
    % m_RdLdy_t is vmasked above. 
    m_RdLdWzr_n = WzrMM_trial * (m_WzrMM_n .* (m_RdLdy_t * t_rt_n));       
    n_RdLdWru_v = WruMM_trial * (n_WruMM_v .* (n_RdLdx_t * v_u_t'));
    n_RdLdWrr_n = WrrMM_trial * (n_WrrMM_n .* (n_RdLdx_t * n_rm1_t'));
    
    if ( do_learn_biases )        
        n_RdLdbx_1 = bxMM_trial * (n_bxMM_1 .* sum(n_RdLdx_t, 2));
        m_RdLdbz_1 = bzMM_trial * (m_bzMM_1 .* sum(m_RdLdy_t, 2));
    else
        n_RdLdbx_1 = zeros(N,1);
        m_RdLdbz_1 = zeros(M,1);
    end           
                 
    if ( dt_o_tau < 1.0 )		% Multiply the dt/\tau factor for dx/dW
        n_RdLdWru_v = dt_o_tau * n_RdLdWru_v;
        n_RdLdWrr_n = dt_o_tau * n_RdLdWrr_n;
        n_RdLdbx_1 = dt_o_tau * n_RdLdbx_1;
    end
    
    n_RdLdx0_c = zeros(N,net.nICs);
    if do_learn_state_init
        n_RdLdx0_c(:,condition_id) = x0MM_trial * (n_x0MM_c(:,condition_id) .* n_RdLdx0_1);
    end
    
    
    if do_recrec_Frobenius_norm_regularizer     
        if isfield(net, 'frobNormRowIdxs')  %  A reasonable hack for now, allows you to choose a submat to regularize.
            frob_row_idxs = net.frobNormRowIdxs;
            frob_col_idxs = net.frobNormColIdxs;
        else
            frob_row_idxs = 1:N;
            frob_col_idxs = 1:N;
        end
        Nfr = length(frob_row_idxs);       
        fc_drFN_tpl = [n_dr0_1(frob_col_idxs) n_dr_t(frob_col_idxs,:)];
        fr_VWrrFN_fc = n_VWrr_n(frob_row_idxs,frob_col_idxs);
        
        
        % This is part 1alpha in the Evernote note.
        % This first part of the R{dLfrob/dtheta} is positive definite.  There is another piece that is not.  It looks like
        %     \alpha \sum_t 2 J_{kl} 2 r'_j(t) r''_j(t) \del x_j(t) / \del J_{mn}  
        % Note that H(theta) is dL/(dJkl dJmn), and the above expression isn't symmetric wrt these indices (k,l and m,n)
        % (keep in mind that \alpha is the frob_factor in the evernote notes.)
        %
        % The (first part) positive definite part is defined as
        %     \alpha 2 d_ik d_jl r'_j(t)^2  
        % That expression for the Hessian translates to an Rv{} operator expression of
        %     \alpha Vkl \sum_t r'_l(t)^2
        % which is what is implemented here.        
        frob_factor = froRR/(2.0*(T+ic_T_add));        
        sdr2_fc = sum(fc_drFN_tpl.^2,2)'; % row
        fr_frohessvFN_fc = frob_factor * 2.0 * fr_VWrrFN_fc .* repmat(sdr2_fc, Nfr, 1);
        n_frohessv_n = zeros(N,N);
        n_frohessv_n(frob_row_idxs, frob_col_idxs) = fr_frohessvFN_fc;
        n_frohessv_n = WrrMM_trial * (n_WrrMM_n .* n_frohessv_n);
        gv = packRNN(net, n_RdLdWru_v, n_RdLdWrr_n + n_frohessv_n, m_RdLdWzr_n, n_RdLdx0_c, n_RdLdbx_1, m_RdLdbz_1);        
    else
        % Pack it up, pack it in, let me begin, I came to win, battle me that's a sin.
        gv = packRNN(net, n_RdLdWru_v, n_RdLdWrr_n, m_RdLdWzr_n, n_RdLdx0_c, n_RdLdbx_1, m_RdLdbz_1);  % see below
    end    
    
    if do_norm_pres_regularizer  % This is NOT the GN approx, the first piece is, and together the full Hessian.
        m_Wrr_n = n_Wrr_n(norm_pres_reg_mask,:);
        m_VWrr_n = n_VWrr_n(norm_pres_reg_mask,:);                
        
        [~, l_prenorms_m] = normify(m_Wrr_n');
        m_prenorms2_1 = (l_prenorms_m.^2)';
        m_prenorm2_res_1 = (m_prenorms2_1 - norm_pres_reg_dv^2);
        
        fac = 2.0 * norm_pres_reg_weight/N_norm_pres_reg;        
        n_gv_norm_pres_n = zeros(N,N);
        n_gv_norm_pres_n(norm_pres_reg_mask,:) = fac * ((2.0 * m_Wrr_n.^2 .* m_VWrr_n) + (repmat(m_prenorm2_res_1, 1, N) .* m_VWrr_n)); 
        
        % This is already mod masked because n_VWrr_n was mod masked above.
        % Package it.
        gv_norm_pres = packRNN(net, zeros(N,V), n_gv_norm_pres_n, zeros(M,N), zeros(N,net.nICs), zeros(N,1), zeros(M,1));
        gv = gv + gv_norm_pres;      
    end
    
    % Add the weight decay and lambda terms.
    gv = gv + wc * (mod_mask .* cost_mask .* v) + lambda * v;
    
    vGv = dot(v,gv);
    
    do_check_GV = 0;
    if vGv < 0
        disp(['vGv: ', num2str(vGv)]);
        do_check_GV = false;
    end    
           
    if do_check_GV && norm(gv) > 0.001 && norm(v) > 0 && rand < 0.01
        disp(['Norm of Gv product: ' num2str(norm(gv)) '.']);
        disp('Explicitly calculating the Gv product.');
        
        % This code is dependent on matching loss!  It's hard to go up
        % to linear portion only, so for cross entropy, I invert.
        % Softmax won't work this way, but I doubt there'd be a problem
        % if both linear and logistic are working (and cross-entropy is
        % correctly defined)        
        EPS = 1e-2;
        theta = net.theta;
        testnetp = net;
        testnetm = net;
        nparams = length(gv);        
        p_G_p = zeros(nparams,nparams);
        m_dX_p = zeros(M,nparams);
        
        %function varargout = rnn_hf_allfun_trace2(net, v_u_t, m_target_t, wc, v, lambda, forward_pass, ...
        %    training_vs_validation, trial_id, optional_params, simdata, ...
        %    do_return_network, do_return_L, do_return_L_grad, ...
        %    do_return_L_GaussNewton, do_return_preconditioner)        
        eval_network = @(net) rnn_hf_allfun2(net, v_u_t, m_target_t, wc, [], [], [], ...
            1, 1, optional_params, simdata, ...
            1, 0, 0, 0, 0);
                                
        % This will not equal the backprop routines if all the parameters aren't being learned.  This is because the hessian is dependent on all
        % parameters.  Note, there is only one example at this layer.  There was an outer loop over examples in the deep net.                
        
        % Note also, that I haven't checked whether or not the regularizations are clean numerically.    Nor have I checked nonlinear outputs.
        xm_abs_diffs = zeros(1, nparams);
        xp_abs_diffs = zeros(1, nparams);
        for i = 1:nparams
            e_i = zeros(nparams,1);
            e_i(i) = 1;                        
            
            theta_i_minus = theta - EPS*e_i;
            theta_i_plus = theta + EPS*e_i;
            
            testnetp.theta = theta_i_plus;
            testnetm.theta = theta_i_minus;
            
            package = eval_network(testnetp);
            forward_pass_p = package{1};
            m_zp_nvm_t = forward_pass_p{3};
            m_zp_t = zeros(M,T);
            m_zp_t(vmask) = m_zp_nvm_t(vmask);
            
            package = eval_network(testnetm);
            forward_pass_m = package{1};
            m_zm_nvm_t = forward_pass_m{3};
            m_zm_t = zeros(M,T);
            m_zm_t(vmask) = m_zm_nvm_t(vmask);
            
            n_xpre_t = forward_pass_m{5};
            n_xpost_t = forward_pass_p{5};
            
            xm_abs_diffs(i) = mean(vec(abs(n_xpre_t - n_x_t)));
            xp_abs_diffs(i) = mean(vec(abs(n_xpost_t - n_x_t)));
            %disp(['minus thing: ' num2str(xm_abs_diffs) '.']);
            %disp(['plus thing: ' num2str(xp_abs_diffs) '.']);
            
            
            switch net.objectiveFunction
                case 'sum-of-squares'
                    m_hprime_1 = ones(M, 1);
                case 'cross-entropy'
                    switch net.layers(end).type
                        case 'logistic'
                            assert ( false, 'Case not implemented yet.');
                        case 'softmax'
                            assert ( false, 'Case not implemented yet.');
                        otherwise
                            disp('Eat shit and die!');
                    end
                otherwise
                    assert ( false, 'stopped');
            end
            
            % This line will be wrong if in a given time slice, there is only one output defined.
            m_dX_p(:,i) = sum(m_zp_t-m_zm_t,2)/(2.0*EPS * length(find(vmask))/M);
        end
        p_G_p = p_G_p + (m_dX_p' * diag(m_hprime_1) * m_dX_p) / M;        % This would be a sum over samples, normally.
        
        Gv = p_G_p*v; 
        gvm = gv -  wc * (mod_mask .* cost_mask .* v) - lambda * v;
        figure; 
        stem(Gv, 'g');
        hold on;
        stem(gvm, 'r');
        figure; stem(xm_abs_diffs, 'b');  % Just looking at magnitude here.
        hold on; stem(-xp_abs_diffs, 'c');
        disp(['Hi! ' num2str(norm(Gv-gvm))]);
        diff = norm(Gv-gvm)/norm(Gv+gvm);
        disp(diff);       
        
        
        [n_GvWru_v, n_GvWrr_n, m_GvWzr_n, n_Gvx0_c, n_Gvbx_1, m_Gvbz_1] = unpackRNN(net, Gv); 
        [n_gvmWru_v, n_gvmWrr_n, m_gvmWzr_n, n_gvmx0_c, n_gvmbx_1, m_gvmbz_1] = unpackRNN(net, gvm); 
    
        % I believe these results are HIGHLY dependent on whether or not the gradient is exploding or vanishing.
        % If the gradient is exploding, this can look very, very ugly.  Makes one wonder how this works at all!
        % If the gradient is vanishing, then this looks very, very good: order 1e-10.
        % Nevertheless, I have noticed that the gradient can still be good order(1e-9), while these look like crap, so there may be a bug, but I'm
        % doubtful.  I just don't understand why the gradient would be more reliable.  Perhaps it is because Gv is a second order calculation.
        
        % To even get a correct answer one time in a network with over a thousand parameters between a
        % numerical and backprop calculation means the main code path is almost certainly implemented correctly (assuming that there are no zeros in
        % v), etc.  Perhaps when the curvature really falls apart is when the network is near a bifurcation, then the step theta + eps (for example)
        % will take the network into a qualitatively new behavior.  Assuming that the backprop routine is not exactly at a bifurcation (set of measure
        % zero hopefully).  So there is no way that backprop and numerical steps will be in alignment then, because it is the finite difference 
        % methodology that is failing.  If this is true, which it must be on a theoretical level, if not a practical level, then why doesn't the
        % actual gradient computation show as bad performance as the Gv calculdation?
        %
        % I tested this explicitly by looking at how a +/- EPS changes the firing rates.  Sometimes the Gv and numerical Gv are still very different
        % despite the fact the that the activitions of the networks are very similar.  So there goes that theory.  Nevertheless, it appears that
        % things look good very often.  I'm putting this down for now.  DCS:12/1/2013
        
        
        disp(['n_Wru_v: ' num2str(mean(vec(abs(n_GvWru_v - n_gvmWru_v))))]);
        disp(['n_ICx_c: ' num2str(mean(vec(abs(n_Gvx0_c - n_gvmx0_c))))]);
        disp(['n_Wrr_n: ' num2str(mean(vec(abs(n_GvWrr_n - n_gvmWrr_n))))]);
        disp(['n_bx_1: ' num2str(mean(vec(abs(n_Gvbx_1 - n_gvmbx_1))))]);
        disp(['m_Wzr_n: ' num2str(mean(vec(abs(m_GvWzr_n - m_gvmWzr_n))))]);
        disp(['m_bz_1: ' num2str(mean(vec(abs(m_Gvbz_1 - m_gvmbz_1))))]);
        
        % Even if the calculation is inaccurate, it should never deliver a negative dot product, if it is correct implemented
        disp(['vnGv: ', num2str(dot(v, Gv))]);
        disp(['vGv: ', num2str(dot(v, gvm))]);
        fprintf('Norm of the difference between Gv products (should be < 1e-9)\n\n');
    end
    
end


%% Return the Outputs
varargout_pre = {};
nouts = 0;
if ( do_return_network )
    if is_training
        varargout_pre{end+1} = 	{n_r_t n_r0_1 n_x0_1 };  % I've cornered myself in here with this rather random ordering.
    else
        varargout_pre{end+1} = 	{n_r_t n_dr_t m_z_t n_r0_1 n_x_t n_x0_1 };  % I've cornered myself in here with this rather random ordering.
    end
    nouts = nouts+1;
end
if ( do_return_L )
    varargout_pre{end+1} = L;
    varargout_pre{end+1} = all_Ls;  %  Tacking this on cuz really useful to monitor.
    nouts = nouts+1;
end
if ( do_return_L_grad )
    varargout_pre{end+1} = grad;
    nouts = nouts+1;
end
if ( do_return_L_GaussNewton )
    varargout_pre{end+1} = gv;
    nouts = nouts+1;
end
if ( do_return_preconditioner )
    varargout_pre{end+1} = precon;
    nouts = nouts+1;
end

varargout_pre{end+1} = simdata;
varargout = {varargout_pre};
