function varargout = mrnn_hf_allfun2(net, v_u_t, m_target_t, wc, v, lambda, forward_pass, ...
    training_vs_validation, trial_id, optional_params, simdata, ...
    do_return_network, do_return_L, do_return_L_grad, ...
    do_return_L_GaussNewton, do_return_preconditioner)
% function varargout = mrnn_hf_allfun2(net, v_u_t, m_target_t, wc, v, lambda, forward_pass, ...
%    training_vs_validation, trial_id, optional_params, simdata, ...
%    do_return_network, do_return_L, do_return_L_grad, ...
%    do_return_L_GaussNewton, do_return_preconditioner)
%
% Written by David Sussillo (C) 2013
%
% This function implements the multiplicative recurrent neural network as described in Sutskever et at. ICML 2011.
% Generating Text with Recurrent Neural Networks.
%
% This function will do it all, that's the only way I know how to keep the code from having multiple implementations.
%
%
% A layer is respect to the weights, so input -> weights -> recurrent  ( layer 1 )
%                                       recurrent -> weights -> recurrent  ( layer 2 )
%                                       recurrent -> weights -> output  ( layer 3 )
%
% net - the network structure.
% v_u_t - the input to the network
%
% m_target_t - the targets for the network
% (can be NaN if undefined, or a numerical values)
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

%% Basic setup

% in hfopt2.m, eval is inputs -> forward_pass -> targets.  This means that the targets may be empty for the forward pass.
if isempty(m_target_t)
    assert ( do_return_network && ~(do_return_L || do_return_L_grad || do_return_L_GaussNewton || do_return_preconditioner ), 'Fucked');
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
if isfield(net, 'doLearnStateInit')
    do_learn_state_init = net.doLearnStateInit;
end

rec_trans_fun = net.layers(2).transFun;
out_trans_fun = net.layers(3).transFun;
rec_deriv_fun = net.layers(2).derivFunAct;
out_deriv_fun = net.layers(3).derivFunAct;

[n_Wru_v, n_Wrr_n, m_Wzr_n, n_bx0_c, n_bx_1, m_bz_1] = unpackMRNN(net, net.theta);
% The update equation has for the recurrent connectivity part:
% x(t) = ... + [n_Wrf_f * diag ( f_Wfu_v * u(t) ) * f_Wfr_n] * r(t-1) + ...
% This boils down to
% \sum_f^F W_rf ( \sum_v^V W_fv u_v(t) ) ( \sum_k^N W_fk r_k(t-1) )
% The sizes of the matries are used (along with rfu in the middle) to identify which matrix is which.
n_Wrf_f = n_Wrr_n.n_W_f;
f_Wfu_v = n_Wrr_n.f_W_v;
f_Wfr_n = n_Wrr_n.f_W_n;


[V,T] = size(v_u_t);		% get relevant dimensions
[M,N] = size(m_Wzr_n);

assert (T > 0, 'fucked' );
assert (M > 0, 'fucked' );
assert (N > 0, 'fucked' );


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
% the right place in state space.  On the other hand, at the end of a trial, we have exactly
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
n_bx0_1 = n_bx0_c(:,condition_id);


if do_use_saved_ics
    assert ( do_use_trial_ics == false, 'ICs are wrong.');
    n_x0_1 = n_sx0_1;
elseif do_use_trial_ics
    assert ( do_use_saved_ics == false, 'ICs are wrong.');
    n_x0_1 = n_tx0_1;
else
    n_x0_1 = n_bx0_1;
end
n_r0_1 = rec_trans_fun(n_x0_1);





%% Value masks.
% The mask for times that have no nans at all.
%tmask = ~isnan(sum(m_target_t,1));
vmask = ~isnan(m_target_t);     % Use logical indexing to allow a single time index with both a value and NaN. DCS:2/15/2012
ntargets = length(find(vmask));
%assert ( ntargets > 0, 'Something wrong here.');
assert ( M > 0, 'Something wrong here.');
% Allow all nans, so we short-circuit some computation below.
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
    f_Wfu_x_u_t = f_Wfu_v * v_u_t;    % A little parallelism
    n_nnoise_t = zeros(N,T);
    if ( noise_sigma > 0.0 )
        n_nnoise_t = noise_sigma * randn(N,T);
    end
    if ( ~isempty(n_addinput_t) )
        n_nnoise_t = n_nnoise_t + n_addinput_t;
    end
    for t = 1:T
        n_Wr_1 = n_Wrf_f * ((f_Wfu_x_u_t(:,t)) .* (f_Wfr_n * n_r_1)); % [n_Wrf_f * f_ diag(f_Wfu_v * v_ut_1) _f * f_Wfr_n] * n_r_1
        n_x_1 = (1.0-dt_o_tau)*n_x_1 + dt_o_tau*( n_Wu_t(:,t) + n_Wr_1 + n_bx_1 + n_nnoise_t(:,t) );
        n_r_1 = rec_trans_fun( n_x_1 );
        n_x_t(:,t) = n_x_1;
        n_r_t(:,t) = n_r_1;
    end
    
    n_dr_t = rec_deriv_fun(n_r_t);
    m_z_t = out_trans_fun( m_Wzr_n * n_r_t + repmat(m_bz_1, 1, T) ); % a little parallelism
    
else
    n_r_t = forward_pass{1};
    n_dr_t = forward_pass{2};
    m_z_t = forward_pass{3};
    n_x_t = forward_pass{5};
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
        L_l2weight = (wc/2.0)*sum((mod_mask .* cost_mask .* net.theta).^2);
        all_Ls(end+1) = L_l2weight;
    end
    L = sum(all_Ls);
end


%% Backprop through time for gradient of RNN weights.
% Backward pass for data, i.e. standard backprop, totally ignoring canonical link, since taking deriv things will
% cancel anyways.
if ( do_return_L_grad )
    
    if ( dt_o_tau < 1.0 )
        n_xdec_n = (1.0-dt_o_tau);	         % as it it were identity times scalar
        f_Wfrt_dec_n = dt_o_tau * n_Wrf_f';  % % transpose this mofo and dt_o_tau
    else
        n_xdec_n = 0;		      % as if it were zero matrix
        f_Wfrt_dec_n = n_Wrf_f';      % transpose this mofo
    end
    
    n_Wrft_f = f_Wfr_n';  % Transpose.   Yikes, this is scary notation.
    n_Wrzt_m = m_Wzr_n';	                 % transpose this mofo
    
    m_dLdy_t = zeros(M,T);
    if ( net.hasCanonicalLink )
        m_dLdy_t(vmask) = m_z_t(vmask) - m_target_t(vmask);
    else
        assert ( false, 'Double check this for the specific case.');
        m_dLdy_t(vmask) = out_deriv_fun(m_z_t(vmask)) .* (m_z_t(vmask) - m_target_t(vmask));
    end
    m_dLdy_t = m_dLdy_t / TxM_vmasked;
    
    % Backward pass.
    n_dLdx_t = zeros(N,T);
    n_dLdx_1 = zeros(N,1);
    
    f_Wfu_x_u_with_0_at_tp1 = f_Wfu_v * [v_u_t zeros(V,1)];  % A little parallism.
    for t = T:-1:1			% backward pass, this t refers to t-1 in dLdx(t-1) = f(dLdx(t)), it's the lhs.
        m_dLdy_1 = m_dLdy_t(:,t);
        n_dLdx_1 = n_xdec_n * n_dLdx_1 + n_dr_t(:,t) .* (n_Wrzt_m * m_dLdy_1 + ...
            n_Wrft_f * (f_Wfu_x_u_with_0_at_tp1(:,t+1) .* (f_Wfrt_dec_n * n_dLdx_1)) ...
            );
        n_dLdx_t(:,t) = n_dLdx_1;
    end
    n_dLdx0_1 = zeros(N,1);
    if ( do_learn_state_init )	 % time is zero
        n_dLdx0_1 = n_xdec_n * n_dLdx_1 + rec_deriv_fun(n_r0_1) .* (n_Wrft_f * (f_Wfu_x_u_with_0_at_tp1(:,1) .* (f_Wfrt_dec_n * n_dLdx_1)) ...
            );
    end
    
    % Now update the the derivatives wrt to the weights. The direct application of the chain rule for partial derivatives does
    % not contain a mean averaging, it's a sum, so there's no mean normalization here. The chain rule is used to get
    % dLdx(t) (that's not normalized by T above) and also to get dLdtheta from dLdx_i * dxi/dtheta
    [n_WruMM_v, n_WrrMM_n, m_WzrMM_n, n_x0MM_c, n_bxMM_1, m_bzMM_1] = unpackMRNNUtils(net, 'domodmask', true);
    n_WrfMM_f = n_WrrMM_n.n_W_f;
    f_WfvMM_v = n_WrrMM_n.f_W_v;
    f_WfrMM_n = n_WrrMM_n.f_W_n;
    
    t_rt_n = n_r_t';			% transpose for speed
    % m_dLdy_t is already vmasked
    m_dLdWzr_n = m_WzrMM_n .* (m_dLdy_t * t_rt_n);
    n_dLdWru_v = n_WruMM_v .* (n_dLdx_t * v_u_t');
    n_rm1_t = [n_r0_1 n_r_t(:,1:end-1)];
    f_alpha_t = f_Wfu_v * v_u_t;
    f_betatm1_t = f_Wfr_n * n_rm1_t;  % t-1
    f_Wrf_x_dLdx_t = n_Wrf_f' * n_dLdx_t;
    n_dLdWrf_f = n_WrfMM_f .*  (n_dLdx_t  * (f_alpha_t .* f_betatm1_t)');
    f_dLdWfv_v = f_WfvMM_v .* ((f_Wrf_x_dLdx_t .* f_betatm1_t) * v_u_t');
    f_dLdWfr_n = f_WfrMM_n .* ((f_Wrf_x_dLdx_t .* f_alpha_t) * n_rm1_t');
    
    if ( do_learn_biases )
        n_dLdbx_1 = n_bxMM_1 .* sum(n_dLdx_t, 2);
        % m_dLdy_t is already vmasked
        m_dLdbz_1 = m_bzMM_1 .* sum(m_dLdy_t, 2);
    else
        n_dLdbx_1 = zeros(N,1);
        m_dLdbz_1 = zeros(M,1);
    end
    
    if ( dt_o_tau < 1.0 )		% Multiply the dt/\tau factor that comes from dx/dW (e.g. r_j(t-1))
        n_dLdWru_v = dt_o_tau * n_dLdWru_v;
        n_dLdWrf_f = dt_o_tau * n_dLdWrf_f;
        f_dLdWfv_v = dt_o_tau * f_dLdWfv_v;
        f_dLdWfr_n = dt_o_tau * f_dLdWfr_n;
        n_dLdbx_1 = dt_o_tau * n_dLdbx_1;
    end
    
    % Pack it up, pack it in.
    n_dLdx0_c = zeros(N,net.nICs);
    n_dLdx0_c(:,condition_id) = n_x0MM_c(:,condition_id) .* n_dLdx0_1;
    n_dLdWrr_n.n_W_f = n_dLdWrf_f;
    n_dLdWrr_n.f_W_v = f_dLdWfv_v;
    n_dLdWrr_n.f_W_n = f_dLdWfr_n;
    grad = packMRNN(net, n_dLdWru_v, n_dLdWrr_n, m_dLdWzr_n, n_dLdx0_c, n_dLdbx_1, m_dLdbz_1);
    
    % Add the weight decay terms.
    grad = grad + wc * (mod_mask .* cost_mask .* net.theta);
    
    do_check_grad = 0;
    if ( do_check_grad && norm(grad) > 0.01 && simdata.id == 3)
        disp(['Norm of backprop gradient: ' num2str(norm(grad)) '.']);
        disp('Numerically checking the gradient created by backprop.');
        %function numgrad = computeNumericalGradient(L, theta)
        % numgrad = computeNumericalGradient(L, theta)
        % theta: a vector of parameters
        % L: a function that outputs a real-number. Calling y = L(theta) will return the
        % function value at theta.
        
        % Initialize numgrad with zeros
        theta = net.theta;
        numgrad = zeros(size(theta));
        EPS = 1e-4;
        
        ngrads = size(theta(:),1);
        
        
        eval_objfun = @(net) mrnn_hf_allfun2(net, v_u_t, m_target_t, wc, [], [], [], 1, 1, optional_params, simdata, 0, 1, 0, 0, 0);
        
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
        
        
        [n_ngWru_v, n_ngWrr_n, m_ngWzr_n, n_ngx0_c, n_ngbx_1, m_ngbz_1] = unpackMRNN(net, numgrad);
        [n_gWru_v, n_gWrr_n, m_gWzr_n, n_gx0_c, n_gbx_1, m_gbz_1] = unpackMRNN(net, grad);
        
        % I believe these results are HIGHLY dependent on whether or not the gradient is exploding or vanishing.
        % If the gradient is exploding, this can look very, very ugly.  Makes one wonder how this works at all!
        % If the gradient is vanishing, then this looks very, very good: order 1e-10.
        % Nevertheless, I have noticed that the gradient can still be good order(1e-9), otherwise it's down to 1e-5 or worse.
        disp(['n_Wru_v: ' num2str(mean(vec(abs(n_ngWru_v - n_gWru_v))))]);
        disp(['n_ICx_c: ' num2str(mean(vec(abs(n_ngx0_c - n_gx0_c))))]);
        disp(['n_Wrr_n.n_W_f: ' num2str(mean(vec(abs(n_ngWrr_n.n_W_f - n_gWrr_n.n_W_f))))]);
        disp(['n_Wrr_n.f_W_v: ' num2str(mean(vec(abs(n_ngWrr_n.f_W_v - n_gWrr_n.f_W_v))))]);
        disp(['n_Wrr_n:.f_W_n ' num2str(mean(vec(abs(n_ngWrr_n.f_W_n - n_gWrr_n.f_W_n))))]);
        disp(['n_bx_1: ' num2str(mean(vec(abs(n_ngbx_1 - n_gbx_1))))]);
        disp(['m_Wzr_n: ' num2str(mean(vec(abs(m_ngWzr_n - m_gWzr_n))))]);
        disp(['m_bz_1: ' num2str(mean(vec(abs(m_ngbz_1 - m_gbz_1))))]);
        
        diff = norm(numgrad-grad)/norm(numgrad+grad);
        disp(diff);
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
    [n_VWru_v, n_VWrr_n, m_VWzr_n, n_vx0_c, n_vbx_1, m_vbz_1] = unpackMRNN(net, v);
    [n_WruMM_v, n_WrrMM_n, m_WzrMM_n, n_x0MM_c, n_bxMM_1, m_bzMM_1] = unpackMRNNUtils(net, 'domodmask', true);
    % If parameters are not modifiable, then R{w} is 0.
    n_VWru_v = n_VWru_v .* n_WruMM_v;
    n_VWrf_f = n_VWrr_n.n_W_f .* n_WrrMM_n.n_W_f;
    f_VWfu_v = n_VWrr_n.f_W_v .* n_WrrMM_n.f_W_v;
    f_VWfr_n = n_VWrr_n.f_W_n .* n_WrrMM_n.f_W_n;
    m_VWzr_n = m_VWzr_n .* m_WzrMM_n;
    %n_vx0_c = n_vx0_c .* n_x0MM_c;   % Handled differently below.
    n_vbx_1 = n_vbx_1 .* n_bxMM_1;    % Also handled below cuz init_mrnn may be wrong.
    m_vbz_1 = m_vbz_1 .* m_bzMM_1;    % ""
    
    n_vx0_1 = n_vx0_c(:,condition_id);
    if do_learn_state_init  % If we learn the IC, then it's a parameter, so R{IC} = vIC.  Else, if the IC is a constant, R{IC} = 0
        n_Rx0_1 = n_vx0_1;
    else
        n_Rx0_1 = zeros(N,1);
    end
    if do_learn_biases  % If we learn the biases, then it's a parameter, so R{b^{x|z}}} = vb^{x|z}.  Else, it's a constant, so R{.} = 0.
        m_vbz_or_zero_1 = m_vbz_1;
        n_vbx_or_zero_1 = n_vbx_1;
    else
        n_vbx_or_zero_1 = zeros(N,1);
        m_vbz_or_zero_1 = zeros(M,1);
    end
    
    n_Rr0_1 = rec_deriv_fun(n_r0_1) .* n_Rx0_1;
    n_Rx_1 = n_Rx0_1;
    n_Rr_1 = n_Rr0_1;
    
    % f1: Forward pass for R operation, so called f1 pass in Schraudolph, giving J_F (v)
    n_rm1_t = [n_r0_1 n_r_t(:,1:end-1)];  % Note the ic first, so there is an index shift below.
    n_VWruu_t = n_VWru_v * v_u_t;	% A little parallelism
    m_VWzrr_t = m_VWzr_n * n_r_t;	% ""
    f_Valpha_t = f_VWfu_v * v_u_t; % ""
    f_alpha_t = f_Wfu_v * v_u_t;  % ""
    f_betatm1_t = f_Wfr_n * n_rm1_t; % ""
    f_Vbetatm1_t = f_VWfr_n * n_rm1_t;
    
    n_Rr_t = zeros(N,T);		% not saving initialization bias
    for t = 1:T
        n_VWruu_1 = n_VWruu_t(:,t);
        f_alpha_1 = f_alpha_t(:,t);
        f_betam1_1 = f_betatm1_t(:,1+t-1);     % watch 1+t-1 cuz of IC on n_rm1_t
        f_Valpha_1 = f_Valpha_t(:,t);
        f_Vbetam1_1 = f_Vbetatm1_t(:,1+t-1);    % watch 1+t-1 cuz of IC on n_rm1_t
        n_Rx_1 = (1.0-dt_o_tau) * n_Rx_1 + ...
            dt_o_tau * (n_vbx_or_zero_1 + n_VWruu_1 + ...               % R-fp for inputs and bias
            n_VWrf_f * (f_alpha_1 .* f_betam1_1) + ...     % R-fp for recurrent,
            n_Wrf_f * ( (f_Valpha_1 .* f_betam1_1) + ...
            f_alpha_1 .* (f_Vbetam1_1 + f_Wfr_n * n_Rr_1) ...
            )  ...
            );
        
        
        %         n_Rx_1 = (1.0-dt_o_tau) * n_Rx_1 + ...
        %                  dt_o_tau * (n_vbx_or_zero_1 + n_VWruu_t(:,t) + ...                             % R-fp for inputs
        %                              n_VWrf_f * (f_alpha_t(:,t) .* f_betatm1_t(:,1+t-1)) + ...     % R-fp for recurrent, watch 1+t-1 cuz of IC on n_rm1_t
        %                              n_Wrf_f * ( (f_Valpha_t(:,t) .* f_betatm1_t(:,1+t-1)) + ...
        %                                          f_alpha_t(:,t) .* (f_Vbetam1_t(:,1+t-1) + f_Wfr_n * n_Rr_1) ...
        %                                        )  ...
        %                              );
        %
        n_Rr_1 = n_dr_t(:,t) .* n_Rx_1;
        
        n_Rr_t(:,t) = n_Rr_1;
    end
    
    % Now the R-backward pass.  Aside from the output layer, it is exactly the same as the normal backward pass, except swapping in R variables.
    % Technically, I should only have one backward pass function and then use it twice, but the abstractions I've built didn't work out that way, so
    % the code is repeated here, using the R variables instead.
    % H_(L.M)
    m_Ry_t = repmat(m_vbz_or_zero_1,1,T) + m_VWzrr_t + m_Wzr_n*n_Rr_t;
    switch net.layers(end).type
        case 'softmax'
            m_Rz_t = m_Ry_t .* m_z_t  - m_z_t .* repmat( sum( m_Ry_t .* m_z_t, 1 ), [M 1] );
        otherwise
            m_dz_t = out_deriv_fun(m_z_t);
            m_Rz_t = m_dz_t .* m_Ry_t;    % H_(L.M) for m_Rz_t
    end
    
    % R backward pass, r1, and putting it altogether.  Should be almost exactly the backprop code above.
    if ( dt_o_tau < 1.0 )
        n_xdec_n = (1.0-dt_o_tau);	         % as it it were identity times scalar
        f_Wfrt_dec_n = dt_o_tau*n_Wrf_f';  % % transpose this mofo, and decay term for continuous system
    else
        n_xdec_n = 0;		                 % as if it were zero matrix
        f_Wfrt_dec_n = n_Wrf_f';  % % transpose this mofo, and decay term for continuous system
    end
    
    n_Wrft_f = f_Wfr_n';  % Transpose.   Yikes, this is scary notation.
    n_Wrzt_m = m_Wzr_n';
    
    
    n_RdLdx_t = zeros(N,T);
    n_RdLdx_1 = zeros(N,1);
    lambda_mu = lambda * mu;
    
    % backprop H_(L.M) J_N v to get Gv
    % J_N v is r1 pass (m_Rz_t)
    % H_(L.M) J_N v = A J_M J_N v is result of R forward pass.
    m_RdLdy_t = zeros(M,T);
    if ( net.hasCanonicalLink )
        m_RdLdy_t(vmask) = m_Rz_t(vmask);
    else
        assert ( net.hasCanonicalLink, 'R{dLdy} not implemented for noncanonical link functions.' );
    end
    m_RdLdy_t = m_RdLdy_t / TxM_vmasked;
    
    f_Wfu_x_u_with_0_at_tp1 = f_Wfu_v * [v_u_t zeros(V,1)];  % A little parallism.
    for t = T:-1:1
        m_RdLdy_1 = m_RdLdy_t(:,t);
        n_RdLdx_1 = n_xdec_n * n_RdLdx_1 + n_dr_t(:,t) .* (n_Wrzt_m * m_RdLdy_1 + ...
            n_Wrft_f * (f_Wfu_x_u_with_0_at_tp1(:,t+1) .* (f_Wfrt_dec_n * n_RdLdx_1)) ...
            );
        
        % Damping, do I have to to anything else to the error function?  DCS 11/22/2013 Assuming this stays the same from the normal RNN case.
        if ( mu > 0 )
            n_RdLdx_1 = n_RdLdx_1 + (lambda_mu * n_Rr_t(:,t));
        end
        n_RdLdx_t(:,t) = n_RdLdx_1;
    end
    n_RdLdx0_1 = zeros(N,1);
    if ( do_learn_state_init )	 % t = 0 here.
        n_RdLdx0_1 = n_xdec_n * n_RdLdx_1 + rec_deriv_fun(n_r0_1) .* (n_Wrft_f * (f_Wfu_x_u_with_0_at_tp1(:,1) .* (f_Wfrt_dec_n * n_RdLdx_1)) ...
            );
        if ( mu > 0 )
            n_RdLdx0_1 = n_RdLdx0_1 + (lambda_mu * rec_deriv_fun(n_r0_1) .* n_Rx0_1);
        end
    end
    
    % Now update the R{} wrt to the weights. The direct application of the chain rule for partial derivatives does
    % not contain a mean averaging, it's a sum, so there's no mean normalization here. The chain rule is used to get
    % dLdx(t) (that's not normalized by T above) and also to get dLdtheta from dLdx_i * dxi/dtheta
    n_WrfMM_f = n_WrrMM_n.n_W_f;
    f_WfvMM_v = n_WrrMM_n.f_W_v;
    f_WfrMM_n = n_WrrMM_n.f_W_n;
    
    t_rt_n = n_r_t';			% transpose for speed
    % m_dLdy_t is already vmasked
    m_RdLdWzr_n = m_WzrMM_n .* (m_RdLdy_t * t_rt_n);
    n_RdLdWru_v = n_WruMM_v .* (n_RdLdx_t * v_u_t');
    f_Wfrt_x_RdLdx_t = n_Wrf_f' * n_RdLdx_t;
    n_RdLdWrf_f = n_WrfMM_f .*  (n_RdLdx_t  * (f_alpha_t .* f_betatm1_t)');
    f_RdLdWfv_v = f_WfvMM_v .* ((f_Wfrt_x_RdLdx_t .* f_betatm1_t) * v_u_t');
    f_RdLdWfr_n = f_WfrMM_n .* ((f_Wfrt_x_RdLdx_t .* f_alpha_t) * n_rm1_t');
    
    if ( do_learn_biases )
        n_RdLdbx_1 = n_bxMM_1 .* sum(n_RdLdx_t, 2);
        m_RdLdbz_1 = m_bzMM_1 .* sum(m_RdLdy_t, 2);
    else
        n_RdLdbx_1 = zeros(N,1);
        m_RdLdbz_1 = zeros(M,1);
    end
    
    if ( dt_o_tau < 1.0 )		% Multiply the dt/\tau factor that comes from dx/dW (e.g. r_j(t-1))
        n_RdLdWru_v = dt_o_tau * n_RdLdWru_v;
        n_RdLdWrf_f = dt_o_tau * n_RdLdWrf_f;
        f_RdLdWfv_v = dt_o_tau * f_RdLdWfv_v;
        f_RdLdWfr_n = dt_o_tau * f_RdLdWfr_n;
        n_RdLdbx_1 = dt_o_tau * n_RdLdbx_1;
    end
    
    % Pack it up, pack it in.
    n_RdLdx0_c = zeros(N,net.nICs);
    if do_learn_state_init
        n_RdLdx0_c(:,condition_id) = n_x0MM_c(:,condition_id) .* n_RdLdx0_1;
    end
    n_RdLdWrr_n.n_W_f = n_RdLdWrf_f;
    n_RdLdWrr_n.f_W_v = f_RdLdWfv_v;
    n_RdLdWrr_n.f_W_n = f_RdLdWfr_n;
    
    gv = packMRNN(net, n_RdLdWru_v, n_RdLdWrr_n, m_RdLdWzr_n, n_RdLdx0_c, n_RdLdbx_1, m_RdLdbz_1);  % see below
    
    % Add the weight decay and lambda terms.
    gv = gv + wc * (mod_mask .* cost_mask .* v) + lambda * v;
    
    vGv = dot(v,gv);
    
    do_check_GV = 0;
    if vGv < 0
        disp(['Warning: individual trial vGv is negative!: ', num2str(dot(v, gv))]);
        do_check_GV = 0;
    end
    
    if ( do_check_GV && norm(gv) > 0.001 && norm(v) > 0 && simdata.id == 3 )
        disp(['Norm of Gv product: ' num2str(norm(gv)) '.']);
        disp('Explicitly calculating the Gv product.');
        
        % This code is dependent on matching loss!  It's hard to go up
        % to linear portion only, so for cross entropy, I invert.
        % Softmax won't work this way, but I doubt there'd be a problem
        % if both linear and logistic are working (and cross-entropy is
        % correctly defined)
        EPS = 1e-4;
        theta = net.theta;
        testnetp = net;
        testnetm = net;
        nparams = length(gv);
        p_G_p = zeros(nparams,nparams);
        m_dX_p = zeros(M,nparams);
        
        %function varargout = mrnn_hf_allfun_trace2(net, v_u_t, m_target_t, wc, v, lambda, forward_pass, ...
        %    training_vs_validation, trial_id, optional_params, simdata, ...
        %    do_return_network, do_return_L, do_return_L_grad, ...
        %    do_return_L_GaussNewton, do_return_preconditioner)
        eval_network = @(net) mrnn_hf_allfun2(net, v_u_t, m_target_t, wc, [], [], [], ...
            1, 1, optional_params, simdata, ...
            1, 0, 0, 0, 0);
        
        % This will not equal the backprop routines if all the parameters aren't being learned.  This is because the hessian is dependent on all
        % parameters.  Note, there is only one example at this layer.  There was an outer loop over examples in the deep net.
        % Note that I haven't checked whether or not the regularizations are clean numerically.    Nor have I checked nonlinear outputs.
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
                    assert ( false, 'fucked');
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
        disp(['Hi! ' num2str(norm(Gv-gvm))]);
        diff = norm(Gv-gvm)/norm(Gv+gvm);
        disp(diff);
        
        [n_GvWru_v, n_GvWrr_n, m_GvWzr_n, n_Gvx0_c, n_Gvbx_1, m_Gvbz_1] = unpackMRNN(net, Gv);
        [n_gvmWru_v, n_gvmWrr_n, m_gvmWzr_n, n_gvmx0_c, n_gvmbx_1, m_gvmbz_1] = unpackMRNN(net, gvm);
        
        % I believe these results are HIGHLY dependent on whether or not the gradient is exploding or vanishing.
        % If the gradient is exploding, this can look very, very ugly.  Makes one wonder how this works at all!
        % If the gradient is vanishing, then this looks very, very good: order 1e-10.
        % Nevertheless, I have noticed that the gradient can still be good order(1e-9), while these look like crap, so I still think there is some
        % kind of bug.
        disp(['n_Wru_v: ' num2str(mean(vec(abs(n_GvWru_v - n_gvmWru_v))))]);
        disp(['n_ICx_c: ' num2str(mean(vec(abs(n_Gvx0_c - n_gvmx0_c))))]);
        disp(['n_Wrr_n.n_W_f: ' num2str(mean(vec(abs(n_GvWrr_n.n_W_f - n_gvmWrr_n.n_W_f))))]);
        disp(['n_Wrr_n.f_W_v: ' num2str(mean(vec(abs(n_GvWrr_n.f_W_v - n_gvmWrr_n.f_W_v))))]);
        disp(['n_Wrr_n:.f_W_n ' num2str(mean(vec(abs(n_GvWrr_n.f_W_n - n_gvmWrr_n.f_W_n))))]);
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
if ( do_return_network )
    varargout_pre{end+1} = 	{n_r_t n_dr_t m_z_t n_r0_1 n_x_t n_x0_1 };  % I've cornered myself in here with this rather random ordering.
end
if ( do_return_L )
    varargout_pre{end+1} = L;
    varargout_pre{end+1} = all_Ls;
end
if ( do_return_L_grad )
    varargout_pre{end+1} = grad;
end
if ( do_return_L_GaussNewton )
    varargout_pre{end+1} = gv;
end
if ( do_return_preconditioner )
    varargout_pre{end+1} = precon;
end

varargout_pre{end+1} = simdata;
varargout = {varargout_pre};
