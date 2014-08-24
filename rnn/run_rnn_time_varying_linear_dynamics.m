function [n_x_tp1, m_z_tp1]  = run_rnn_time_varying_linear_dynamics(net, fp_struct, n_x0_1, nsteps, varargin)
% function [n_x_tp1, m_z_tp1]  = run_rnn_linear_dynamics(net, fp_struct, n_x0_1, nsteps)
%
% n_xfp_1 is the fixed point
% n_x0_1 is the initial condition for the simulation


optargin = size(varargin,2);
eigs_to_zero = [];
for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
        case 'eigstozero'
            eigs_to_zero = varargin{i+1};  % idxs sorted by real part for continouous    
        otherwise
            assert ( false, [' Variable argument ' varargin{i} ' not recognized.']);
    end
end


dt_o_tau = net.dt / net.tau;
trans_fun = net.layers(2).transFun;
deriv_trans_fun = net.layers(2).derivFunAct;
N = net.layers(2).nPost;
[~, n_Wrr_n, m_Wzr_n, ~, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);
allfp = [fp_struct.FP];


% Now run the linearized system.  
n_x_t = zeros(N,nsteps+1);
n_x_1 = n_x0_1;
n_x_t(:,1) = n_x0_1;
for t = 2:nsteps+1
        
    fp_minus_x = bsxfun(@minus, allfp, n_x_1);
    fp_x0_dist = sqrt(sum(fp_minus_x.^2, 1));
    [~, midx] = min(fp_x0_dist);
    fpclose = fp_struct(midx);
    % Zero order.
    n_xfp_1 = fpclose.FP;    
    n_rfp_1 = trans_fun(n_xfp_1);
    n_Fxfp_1 = -n_xfp_1 + n_Wrr_n * n_rfp_1 + n_bx_1;   % Tau is covered below.
    
    % First order.
    %R = fpclose.eigenVectors;
    %L = fpclose.leftEigenVectors;    
    %eigsall = fpclose.eigenValues;
    %n_jac_n = real(R* diag(eigsall) * L);
    n_drfp_1 = deriv_trans_fun(n_rfp_1);  % Derivatives of the nonlinearity (e.g. tanh)
    nr_drfp_n = ones(N,1)*n_drfp_1';		% r for redundant dimension, so each row is a duplicate
    n_jac_n = -eye(N) + n_Wrr_n .* nr_drfp_n;   % Tau is covered below.
         
    % Run the perturbative dynamics.
    n_px_1 = n_x_1 - n_xfp_1;
    n_pxdot_1 = n_jac_n*n_px_1 + n_Fxfp_1;
    n_px_1 = n_px_1 + dt_o_tau * n_pxdot_1; 
    
    % Put it all back together.
    n_x_1 = n_xfp_1 + n_px_1;
    n_x_t(:,t) = n_x_1;
end

n_x_tp1 = n_x_t;
m_z_tp1 = net.layers(3).transFun( bsxfun(@plus, m_Wzr_n * trans_fun(n_x_tp1), m_bz_1) );

end

