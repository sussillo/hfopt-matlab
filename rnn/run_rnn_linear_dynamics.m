function [n_x_tp1, m_z_tp1]  = run_rnn_linear_dynamics(net, n_xfp_1, v_input_1, n_x0_1, nsteps, varargin)
% function [n_x_tp1, m_z_tp1]  = run_rnn_linear_dynamics(net, n_xfp_1, n_x0_1, nsteps, varargin)
%
% n_xfp_1 is the fixed point
% n_x0_1 is the initial condition for the simulation

if isempty(v_input_1)
    v_input_1 = 0;  % Will work in all odd circumstances.
end

do_use_zero_order = false; % include F(xp) as "input" to the linear dynamical system.
optargin = size(varargin,2);
eigs_to_zero = [];
A = [];
v_inputs_t = [];
do_time_dependent_input = false;
for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
        case 'eigstozero'
            eigs_to_zero = varargin{i+1};  % idxs sorted by real part for continouous
        case 'A'
            A = varargin{i+1};  % just supply the matrix itself
        case 'dousezeroorder'
            do_use_zero_order = varargin{i+1};
        case 'inputs'   
            % Add a time-dependent input to the perturbative system.  One can always add inputs to a linear system, 
            % but you should be careful to make sure it makes sense in the context of a linearization of a nonlinear system.  The main point is that
            % in the linearization of a nonlinear system, the inputs fall out: \del F(x) \del x.  For example,
            % if you wanted to analyze the behavior of a nonlinear system with respect to a time-varying input, then you'd have to relinearize for
            % each static input to the nonlinear system.  
            
            % Having given all of these warnings:  if you simply want to see how the linear system created by linearizing the nonlinear
            % system behaves with respect to some time-varying input, go ahead.  Just be careful on how you interpret the results.
            
            % Finally, note that the input used to create the zero order term, v_input_1, is left alone.  Since it really doesn't make sense to have a
            % time varying input in terms of understanding the nonlinear system, v_input_1, is left as is and the v_input_t is inserted into the for
            % loop over time.
            v_inputs_t = varargin{i+1};
        otherwise
            assert ( false, [' Variable argument ' varargin{i} ' not recognized.']);
    end
end

if ~isempty(v_inputs_t)
    do_time_dependent_input = true;
end
if do_use_zero_order
    assert ( do_time_dependent_input == false, 'Not sure this makes sense.  Think through before using.');
end


trans_fun = net.layers(2).transFun;

n_rfp_1 = trans_fun(n_xfp_1);
dt_o_tau = net.dt / net.tau;
N = net.layers(2).nPost;
[n_Wrv_v, n_J_n, m_Wzr_n, ~, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);

% Get zero order term.
if net.dt == net.tau
    n_Fxp_1 = n_J_n*n_rfp_1 + n_bx_1 + n_Wrv_v * v_input_1;     % Input is part of the F()
else
    n_Fxp_1 = -n_xfp_1 + n_J_n*n_rfp_1 + n_bx_1 + n_Wrv_v * v_input_1;  % Input is part of the F()
end

% Define the Jacobian matrix for the system.
n_drfp_1 = net.layers(2).derivFunAct(n_rfp_1);
nr_drfp_n = ones(N,1)*n_drfp_1';		% dxdoti/dxj = -dij + r'_j J_ij, so r' duplicate in rows
if isempty(A) 
    if ( dt_o_tau < 1.0 )
        % This is what the jacobian of the implied continuous time system differential equation looks like.
        n_jac_n = -eye(N) + n_J_n .* nr_drfp_n;
        
        if ~isempty(eigs_to_zero)
            [V,D] = eig(n_jac_n);
            d = diag(D);
            [d, sidxs] = sort(real(d), 'descend');   % Sort by stability for continuous system.
            V = V(:,sidxs);
            dz = d;
            dz(eigs_to_zero) = 0.0;
            
            n_jac_n = real(V*diag(dz)*pinv(V));  % This real() may be necessary if someone zeros out only one CCP.
        end
    else					% should this be the jacobian of the difference? -DCS:2011/10/27
        assert ( false, 'Not implemented yet.');
    end
else
    n_jac_n = A;
end

% Setup the perturbation initial conditions.
n_px0_1 = n_x0_1 - n_xfp_1;

% Now run the linearized system.  
n_px_t = zeros(N,nsteps);
n_px_1 = n_px0_1;
for t = 1:nsteps
    n_pxdot_1 = n_jac_n*n_px_1;  % no bias because it's never perturbed by input or x.
    n_px_1 = n_px_1 + dt_o_tau * (n_pxdot_1 + do_use_zero_order * n_Fxp_1);
    if do_time_dependent_input
       n_px_1 = n_px_1 + dt_o_tau * n_Wrv_v * v_inputs_t(:,t); 
    end
    n_px_t(:,t) = n_px_1;
end

n_x_tp1 = bsxfun(@plus, [n_px0_1 n_px_t], n_xfp_1);  % put in initial condition
m_z_tp1 = net.layers(3).transFun(bsxfun(@plus, m_Wzr_n * trans_fun( n_x_tp1 ), m_bz_1));
