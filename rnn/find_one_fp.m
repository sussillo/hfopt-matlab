function [q n_gradq_1 n_hessq_n] = find_one_fp(n_x_1, net, const_input, fun_tol, do_topo_map)
% function [L n_gradq_1 n_hessq_n] = find_one_fp(n_x_1, net, const_input, fun_tol, do_topo_map)
%
% Compute the function q(x) = 0.5 |F(x)|^2.  Compute also the derivative, \frac{\partial q}{\partial x_i} and the Gauss Newton approximation to the
% Hessian matrix, \sum_k \frac{\partial F_k}{\partial x_i} \frac{\partial F_k}{\partial x_j}
%
% The values are computed at n_x_1.  The network structure, net, contains all the relevant parameters.  The const_input is whether or not the fixed
% point we're search for is input dependent.  The fun_tol variable is the function tolerance, which we'll manually stop the optimizatio after if we're
% looking for iso-speed contours (do_topo_map = true).


% Separating the continuous time and discrete time cases isn't really necessary.  But every now and then I'll make an advance in the code and
% introduce a bug because things will be ever so slightly different between the two cases.  So I'm keeping them separate for now.  DCS 07/03/2012.

N = net.layers(2).nPost;
dt_o_tau = net.dt / net.tau;
dt = net.dt;
tau = net.tau;
n_r_1 = net.layers(2).transFun(n_x_1);
[n_Wru_v, n_Wrr_n, ~, ~, n_bx_1, ~] = unpackRNN(net, net.theta);

if dt_o_tau < 1.0  % Continuous time equations.  Compute F(x) directly.
    
    % \tau \dot{x} = F(x) = -x + J r + B u + b^x
    
    n_Fx_1 = -n_x_1 + n_Wrr_n * n_r_1 + n_bx_1;
    if ( ~isempty(const_input) )
        n_Fx_1 = n_Fx_1 + (n_Wru_v * const_input);
    end        
       
    if tau ~= 1.0  % If tau is 1.0 and this check fails, that's OK.  I'm just trying to reduce numerical error wherever I find it.
        n_Fx_1 = n_Fx_1 / tau;
    end
        
else  % For the discrete case, we imagine F(x) as G(x(t+1)) - G(x(t)), where G(x) define the discrete update equations.
    assert ( tau == dt, 'No tau or dt for discrete systems.');  % Identically should be OK.
    % Compute x(t+1)
    % x[t+1] = G(x) = J r[t] + B u[t+1] + b^x
    
    n_x1_1 = n_Wrr_n * n_r_1 + n_bx_1;
    if ( ~isempty(const_input) )
        n_x1_1 = n_x1_1 + n_Wru_v * const_input;
    end    
    n_Fx_1 = n_x1_1 - n_x_1;
end

% Compute q(x)
q = 0.5*(n_Fx_1'*n_Fx_1);

% The gradient and the Gauss Netwton matrix.
n_gradq_1 = zeros(N,1);  % Fall through case sets these to zero, which stops the optimization.
n_hessq_n = zeros(N,N);  % Fall through case sets these to zero, which stops the optimization.

if ~do_topo_map || (do_topo_map && q >= fun_tol) 
    n_dr_1 = net.layers(2).derivFunAct(n_r_1);  % Derivatives of the nonlinearity (e.g. tanh)
    nr_dr_n = ones(N,1)*n_dr_1';		% r for redundant dimension, so duplicate in rows
        
    if dt_o_tau < 1.0  % Continuous time.
        n_Fjac_n = -eye(N) + n_Wrr_n .* nr_dr_n;
        if tau ~= 1.0  % If tau is 1.0 and this check fails, that's OK.  I'm just trying to reduce numerical error wherever I find it.
            n_Fjac_n = n_Fjac_n / tau;
        end
        
        % I implemented the Hessian for the continuous time case to be extra sure it didn't matter.  It doesn't.
        % n_ddr_1 = -2 * n_r_1 .* n_dr_1;
        % F_x_d2F_o_dxdx = diag((n_Fx_1' * n_Wrr_n)' .* n_ddr_1);
        
    else   % Discrete time.
        n_Fjac_n = -eye(N) + n_Wrr_n .* nr_dr_n;  % Note this is the Jacobian of F(t) = G(t) - G(t-1), not the Jacobian of G.
        
        % No need to implement.
        % F_x_d2F_o_dxdx = zeros(N,N);
        % assert ( false, 'Not implemented yet.');
    end
    
    n_gradq_1 = (n_Fx_1' * n_Fjac_n)';
    
    n_hessq_n = n_Fjac_n'*n_Fjac_n;                 		% This is the so-called Gauss Newton Approximation, skipping the second part.
    %n_hessq_n = n_Fjac_n'*n_Fjac_n + F_x_d2F_o_dxdx;		% The Real McCoy.
end








end