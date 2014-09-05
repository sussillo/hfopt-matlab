function varargout = dn_hf_allfun2(net, v_input_s, m_target_s, wc, v, lambda, pregen_forward_pass, ...
    training_vs_validation, trial_id, optional_params, simdata, ...
    do_return_network, do_return_L, do_return_L_grad, ...
    do_return_L_GaussNewton, do_return_preconditioner, varargin)
%
% Standard deep net for classification, with dropping optional.
%

% A layer is respect to the weights, so input -> weights -> output  ( layer 1 )
%                                       input -> weights -> output  ( layer 2 )
%                                       input -> weights -> output  ( layer 3 )
%                                       input -> weights -> output  ( layer 4 )
% is a four layer network according to this definition, cuz there are four transformations.


% Not exactly hack, but non-standard for sure.  This routine not only conforms to the hfopt.m API for returning all the
% important optimization objects, but it also will return extra stuff, which can be returned to save time when doing the
% learn dynamics experiments.  
%
% do_save_hp = false; do_save_dEdx = false; dEdx_saved = [];
% 
% optargin = size(varargin,2);
% for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
%     switch varargin{i}
%         case 'dosavehp'
%             do_save_hp = varargin{i+1};
%         case 'dosavedEdx'
%             do_save_dEdx = varargin{i+1};
%         case 'dEdx'
%             dEdx_saved = varargin{i+1};
%         otherwise
%             assert( false, 'Don''t recognize option.');
%     end
% end

[~,S] = size(v_input_s);

[Wu, bu] = unpackDN(net, net.theta);	% What's the 'u' for?  Keep the same from Martens.


nlayers = net.nlayers;
layers = net.layers;

% TvV_T = 1;
% TvV_V = 2;
% is_learning = ~do_return_L;


npre = zeros(1,nlayers);
npost = zeros(1,nlayers);
for i = 1:nlayers
    npre(i) = net.layers(i).nPre;
    npost(i) = net.layers(i).nPost;
end

vmask = ~isnan(m_target_s);     % Use logical indexing to allow a single time index with both a value and NaN. DCS:2/15/2012
%ntargets = length(find(vmask));
%assert ( ntargets > 0, 'Something wrong here.');


%   yp1 cuz first layer isn't saved, so p1 implies counting from input.
if ( isempty(pregen_forward_pass) )    
    n_yim1_s = v_input_s;
    n_yi_s = cell(1,nlayers);
    for i = 1:nlayers			% doesn't include the input, 1 is the first hidden layer              
        trans_fun = layers(i).transFun;
        n_yi_s{i} = trans_fun(Wu{i} * n_yim1_s + repmat(bu{i}, [1 S]));        
        n_yim1_s = n_yi_s{i};
    end
    
    assert ( ~do_return_L_GaussNewton, 'Something is wrong.');
else
    v_input_s = pregen_forward_pass{1};
    n_yi_s = cell(1,nlayers);
    for i = 1:nlayers			% doesn't include the input, 1 is the first hidden layer
        n_yi_s{i} = pregen_forward_pass{i+1};
    end
end


if ( do_return_L )
    % XXX I'm not dividing by M here, even though I do the RNN script.
    % Value masking with nans.
    msv_target_1 = m_target_s(vmask);  % need not be contiguous in time.
    msv_yout_1 = n_yi_s{end}(vmask);   % ""
    
    all_Js = [];
    
    switch net.objectiveFunction
        case 'cross-entropy' 
            switch net.layers(end).type
                case 'logistic'
                    J_output = -sum(sum(msv_target_1 .* log(msv_yout_1+realmin) + (1 - msv_target_1).*log(1-msv_yout_1+realmin)));
                case 'softmax'
                    J_output = -sum(sum(msv_target_1 .* log(msv_yout_1+realmin)));
                otherwise
                    disp('Eat shit and die.');
            end
        case 'sum-of-squares'
            J_output = (1.0/2.0) * sum( sum( (msv_target_1 - msv_yout_1).^2 ));
        otherwise
            disp('Eat shit and die.');
    end
    all_Js(end+1) = J_output;
    
    % Weight mask determines which synaptic strengths have a cost.
    J_L2weight = (wc/2.0)*sum((net.costMask .* net.theta).^2);
    all_Js(end+1) = J_L2weight;
    
    J = sum(all_Js);
end

% Backward pass for data, i.e. standard backprop.
if do_return_L_grad || do_return_preconditioner %|| (do_return_L_GaussNewton && net.simParams.doTrueHv)
    
    assert ( net.hasCanonicalLink, 'We require canonical link functions.' );
    
    % Back prop ( y = h(x) )
    m_error_s = zeros(npost(end),S);
    m_error_s(vmask) = m_target_s(vmask) - n_yi_s{end}(vmask);
    %m_error_s = m_target_s - n_yi_s{end};
    n_dEdx_s{nlayers} = -m_error_s;  	% also known as delta, assumes canonical link.
    n_dEdxi_s = -m_error_s;
    n_hp_s = cell(1, nlayers);
    for i = nlayers:-1:2
        n_hp_s{i-1} = net.layers(i-1).derivFunAct(n_yi_s{i-1});
        n_dEdx_s{i-1} = (Wu{i}' * n_dEdxi_s) .* n_hp_s{i-1};
        n_dEdxi_s = n_dEdx_s{i-1};
    end
    
    if ( do_return_L_grad )		% now make the gradient of the objective function
        
        n_yim1_s = v_input_s;
        n_dEdW_n = cell(1,nlayers);
        n_dEdb_1 = cell(1,nlayers);
        for i = 1:1:nlayers		% no need to go backwards here.
            n_dEdW_n{i} = (n_dEdx_s{i} * n_yim1_s'); % xxx not a square, notation don't help
            n_dEdb_1{i} = sum(n_dEdx_s{i},2);
            n_yim1_s = n_yi_s{i};
        end
        
        % Pack it up.
        grad = packDN(net, n_dEdW_n, n_dEdb_1);
        % Add the weight decay terms.
        grad = grad + wc * (net.costMask .* net.theta);
            
        do_check_grad = 0;
        if ( do_check_grad )
            disp('Numerically checking the gradient created by backprop.');
            %function numgrad = computeNumericalGradient(J, theta)
            % numgrad = computeNumericalGradient(J, theta)
            % theta: a vector of parameters
            % J: a function that outputs a real-number. Calling y = J(theta) will return the
            % function value at theta.
            
            % Initialize numgrad with zeros
            theta = net.theta;
            numgrad = zeros(size(theta));
            EPS = 1e-4;
            
            ngrads = size(theta(:),1);
                       
            %forward_pass = {v_uperturbed_s n_yi_s{:}}; % may not be practical down the road.
            
            simdata.DNForwardPass = {};
            eval_objfun_dn = @(net) dn_hf_allfun2(net, v_input_s, m_target_s, wc, [], [], [], training_vs_validation, trial_id, optional_params, simdata, 0, 1, 0, 0, 0);
            for i = 1:ngrads
                e_i = zeros(ngrads,1);
                e_i(i) = 1;
                
                theta_i_plus = theta + EPS*e_i;                               
                testnetp = net;                
                testnetp.theta = theta_i_plus;                
                package = eval_objfun_dn(testnetp);
                E_p = package{1};
                
                theta_i_minus = theta - EPS*e_i;
                testnetm = net;
                testnetm.theta = theta_i_minus;
                package = eval_objfun_dn(testnetm);
                E_m = package{1};
                numgrad(i) = (E_p - E_m)/(2.0*EPS);
                             
                if mod(i,1000) == 0
                    disp(i);
                end
            end
            
            diff = norm(numgrad-grad)/norm(numgrad+grad);
            disp(diff);
            fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');
            
        end
    end
    
    if ( do_return_preconditioner )		% wc, based on Martens' code.
        assert ( false, 'Need to check correctness.');  % Not being used right now, but algorithms should still work.
        % I'm not sure why we square first and then multiply, instead of multiply first and then square, but I'm
        % following Martens' code as well as possible. -DCS:2011/08/17
        n_yim1_s = v_input_s;
        n_dEdW2_n = cell(1,nlayers);
        n_dEdb2_1 = cell(1,nlayers);
        for i = 1:1:nlayers		% no need to go backwards here.
            n_dEdxi_sqr_s = n_dEdx_s{i}.^2;
            n_dEdW2_n{i} = (n_dEdxi_sqr_s * (n_yim1_s.^2)'); % xxx not a square, notation don't help
            n_dEdb2_1{i} = sum(n_dEdxi_sqr_s,2);
            n_yim1_s = n_yi_s{i};
        end
        
        % Pack it up.
        grad2 = packDN(net, n_dEdW2_n, n_dEdb2_1);
        % Add the weight decay terms and lambda.
        alpha = 3.0/4.0;
        precon = (grad2 + (ones(size(grad2)) * lambda) + (wc * net.costMask) ).^alpha; % xxx User beware: didn't ever double check this!
        
    end
    
end

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

% I'm reasonably sure this is 100% but I'd like to check it numerically.  The problem is I can't find a reference on
% it and Martens' code is "not the cleanest" wrt to this.
if ( do_return_L_GaussNewton && ~net.simParams.doTrueHv)
    
    [VWu, Vbu] = unpackDN(net, v);		% What's the 'u' for?  Keep the same from Martens.
    
    assert ( net.hasCanonicalLink, 'We require canonical link functions.' );
    
    
    
    % f1: Forward pass for R operation, so called f1 pass in Schraudolph, giving J_F (v)
    n_yim1_s = v_input_s;
    n_Ryim1_s = zeros(net.layers(1).nPre,S);			% W * Ry don't count in first layer
    n_hp_s = cell(1,nlayers);
    for i = 1:nlayers			% up to output (linear)
        n_Rxi_s = Wu{i} * n_Ryim1_s + VWu{i} * n_yim1_s + repmat(Vbu{i}, [1 S]);
        
        switch net.layers(i).type
            case 'softmax'  % Will break below in backward pass if this is not at the end.
                n_Ryi_s = n_Rxi_s .* n_yi_s{i}  - n_yi_s{i} .* repmat( sum( n_Rxi_s .* n_yi_s{i}, 1 ), [npost(end) 1] );
            otherwise
                n_hp_s{i} = net.layers(i).derivFunAct(n_yi_s{i});  % optimized, saved from backward pass.
                n_Ryi_s = n_hp_s{i} .* n_Rxi_s;    % H_(L.M) for n_Ryi_s{end} is there from n_derivs_s{end}
        end
        n_Ryim1_s = n_Ryi_s;
        n_yim1_s = n_yi_s{i};
    end           

    
    % R backward pass, r1
    n_RdEdx_s{nlayers} = zeros(npost(end), S);
    n_RdEdx_s{nlayers}(vmask) = n_Ryi_s(vmask);
    n_RdEdxi_s = n_RdEdx_s{nlayers};
    %n_RdEdx_s{nlayers} = n_Ryi_s;
    %n_RdEdxi_s = n_Ryi_s;
    for i = nlayers:-1:2
        n_RdEdx_s{i-1} = (Wu{i}' * n_RdEdxi_s) .* n_hp_s{i-1};  % why not also (VWu{i}' * n_dEdxi_s{i} ) .* n_hp_s{i-1} ? DCS:5/3/2013
        n_RdEdxi_s = n_RdEdx_s{i-1};
    end
    
    % R gradient, put it all together
    n_yim1_s = v_input_s;
    n_VWgrad_n = cell(1,nlayers);
    n_Vbgrad_1 = cell(1,nlayers);
    for i = 1:nlayers
        % In full Hv implementation, we have an additional term (see pg. 256 in Jordan).  Not sure if should be here or not.  I don't think so because
        % the numerical checker was correct for the linear and logistic cases, and Schraudolph says it's another backprop.  So dEdx shouldn't enter
        % in here (it doesn't in GN passes above wheras it does in Hv passes below).  Note I'm' not sure 100%, though.  
        n_VWgrad_n{i} = (n_RdEdx_s{i} * n_yim1_s'); 
        n_Vbgrad_1{i} = sum(n_RdEdx_s{i},2);
        n_yim1_s = n_yi_s{i};
    end
    
    % Pack it up.
    gv = packDN(net, n_VWgrad_n, n_Vbgrad_1);
    % Add the weight decay terms.
    gv = gv + wc * (net.costMask .* v);
    % Add the lambda regularizer.
    gv = gv + lambda * v;
    
    
    do_check_GV = 0;
    if ( do_check_GV )
        disp('Explicitly calculating the Gv product.'); %#ok<*UNRCH>
        
        % This code is dependent on matching loss!  It's hard to go up
        % to linear portion only, so for cross entropy, I invert.
        % Softmax won't work this way, but I doubt there'd be a problem
        % if both linear and logistic are working (and cross-entropy is
        % correctly defined)        
        EPS = 1e-4;
        theta = net.theta;
        nparams = length(gv);        
        M = npost(end);
        testnetp = net;
        testnetm = net;
        p_G_p = zeros(nparams,nparams);
        m_dX_p = zeros(M,nparams);
        for s = 1:S
            for i = 1:nparams                
                e_i = zeros(nparams,1);
                e_i(i) = 1;
                theta_i_minus = theta - EPS*e_i;
                theta_i_plus = theta + EPS*e_i;
                
                testnetp.theta = theta_i_plus;
                testnetm.theta = theta_i_minus;
                
                dEdw_p = dn_hf_allfun(testnetp, v_input_s(:,s), m_target_s(:,s), wc, [], [], [], training_vs_validation, trial_id(s), optional_params, 1, 0, 0, 0, 0);
                dEdw_m = dn_hf_allfun(testnetm, v_input_s(:,s), m_target_s(:,s), wc, [], [], [], training_vs_validation, trial_id(s), optional_params, 1, 0, 0, 0, 0);
                Y0 = dn_hf_allfun(net, v_input_s(:,s), m_target_s(:,s), wc, [], [], [], training_vs_validation, trial_id(s), optional_params, 1, 0, 0, 0, 0);
                
                m_Yp_1 = dEdw_p{end};
                m_Ym_1 = dEdw_m{end};
                m_Y0_1 = Y0{end};
                
                switch net.objectiveFunction
                    case 'sum-of-squares'
                        m_hprime_1 = ones(M, S);
                    case 'cross-entropy'
                        switch net.layers(end).type
                            case 'logistic'
                                %m_fac_s = sqrt((-m_Y0_s.^2  + 2*m_Y0_s.*m_target_s - m_target_s)./(m_Y0_s.*(1-m_Y0_s)).^2);
                                % Correct factor for dydtheta
                                
                                m_hprime_1 = m_Y0_1.*(1-m_Y0_1);
                                m_Xp_1 = log ( m_Yp_1 ./ (1 - m_Yp_1));
                                m_Xm_1 = log ( m_Ym_1 ./ (1 - m_Ym_1));
                            case 'softmax'
                                1;
                            otherwise
                                disp('Eat shit and die!');
                        end
                    otherwise
                        assert ( false, 'fucked');
                end
                
                m_dX_p(:,i) = (m_Xp_1-m_Xm_1)/(2.0*EPS);                
            end           
            
            p_G_p = p_G_p + (m_dX_p' * diag(m_hprime_1) * m_dX_p);
            
        end
        p_G_p = p_G_p;
        
        Gv = p_G_p*v;
        gvm = gv -  wc * (net.costMask .* v) -  lambda * v;
        
        disp(['Hi! ' num2str(norm(Gv-gvm))]);
        
        fprintf('Norm of the difference between Gv products (should be < 1e-9)\n\n');
            
    end           
end


%% Return the outputs
varargout_pre = {};
if ( do_return_network )
    varargout_pre{end+1} = {v_input_s n_yi_s{:}}; % may not be practical down the road.
end
if ( do_return_L )
    varargout_pre{end+1} = J;
    varargout_pre{end+1} = all_Js;  %  Takcking this on cuz realy useful to monitor.
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

