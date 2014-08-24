function [ds, npos, Vs, Ls] = get_linear_stability_fp(net, n_x0_1, with_eigenvectors)
% function [ds, npos, Vs, Ls] = get_linear_stability(net, n_x0_1, with_eigenvectors)
%
% A function to automate the linear stability analysis around the point n_x0_1.  
% 
% INPUTS
% net - the HF network structure, storing all the parameters.
% n_x0_1 - the point around which the linear stability analysis is done.
% with_eigenvectors - boolean, as to whether or not to return the lefts and the rights.
%
% OUTPUTS
% ds - eigenvalues of Jacobian
% npos - number of unstable eigenvalues for either continuous or discrete linearized system
% Vs - right eigenvectors, returned as empty if with_eigenvectors is false 
% Ls - left eigenvectors , returned as empty if with_eigenvectors is false

% Get relevant params from the network structure.
n_r0_1 = net.layers(2).transFun(n_x0_1);
dt_o_tau = net.dt / net.tau;
N = net.layers(2).nPost;
[~, n_J_n, ~, ~, ~, ~] = unpackRNN(net, net.theta);


% Define the Jacobian matrix for the system. 
n_dr0_1 = net.layers(2).derivFunAct(n_r0_1);
nr_dr0_n = ones(N,1)*n_dr0_1';       % r for redundant, so duplicate in rows

if ( dt_o_tau < 1.0 )   
    % This is what the jacobian of the implied continuous time system differential equation looks like.

    %dxdoti/dxj = -dij + J_ij r'_j, so r' duplicate in rows
    n_jac_n = -eye(N) + n_J_n .* nr_dr0_n;    
else    
    n_jac_n = n_J_n .* nr_dr0_n;    
end

% Now get the eigenvalues and eigenvectors.
if with_eigenvectors
    [V, D] = eig(n_jac_n);
    d = diag(D);
    if ( dt_o_tau < 1.0 )
        [~, idxs] = sort(real(d), 'descend');
        Vs = V(:,idxs);
        ds = d(idxs);
        npos = sum( real(ds) > 0 );
    else
        [~, idxs] = sort(abs(d), 'descend');
        Vs = V(:,idxs);
        ds = d(idxs);
        npos = sum( abs(ds) > 1 );
    end
    Ls = pinv(Vs);  % Get the left eigenvectors as well.
    
    % Consistently orient all the vectors, this comes up enough to put it in
    % the subroutine.
    ref_vec = ones(N,1);
    for i = 1:N
        if dot(Vs(:,i), ref_vec) < 0.0
            Vs(:,i) = -Vs(:,i);   
            Ls(i,:) = -Ls(i,:);   % If you do it for the right, you must do it for the left.
        end
    end
else
    d = eig(n_jac_n);
    Vs = [];
    Ls = [];
    if ( dt_o_tau < 1.0 )
        [~, idxs] = sort(real(d), 'descend');
        ds = d(idxs);
        npos = sum( real(ds) > 0 );
    else
        [~, idxs] = sort(abs(d), 'descend');
        ds = d(idxs);
        npos = sum( abs(ds) > 1 );
    end
end
