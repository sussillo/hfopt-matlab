function [fp_struct, fpd] = find_many_fixed(net, niters, xmeans, epsilon, max_eps, fval_tol, varargin)
% function [fp_struct, fpd] = find_many_fixed(net, niters, xmeans, epsilon, max_eps, fval_tol, varargin)
%
% Front end to finding fixed points for the Hessian Free optimization code.
%
% net - the matlab structure of the network being analyzed
% niters - number of fixed points to find.
% xmeans - an array of initial conditions for the optimizations, typically taken from network trajectories
% epsilon - add gaussian noise with this standard deviation to a given initial condition
% max_eps - the maximum amount of noise to add before deciding to exit or start over
% fval_tol - the minimum value of the fixed point we'll accept.  This is useful for fine control of finding iso-speed contours or extremely accurate
% fixed points.

display = 'off';			% 'off or 'iter', for example.
tolx = 1e-16;               % Perhaps overly small, but it's cautious.
tolfun = 1e-16;             % Perhaps overly small, but it's cautious.

eps_factor = 2.0;
const_input = [];
do_topo_map = true;  % Don't worry about whether the speed is a true local minima, just find low speed areas.
optargin = size(varargin,2);
do_bail = 1;				% bail after we hit the max epsilon and still fail.
opt_max_iters = 10000;
stability_analysis = 'full';  
for i = 1:2:optargin
    switch varargin{i}
        case 'constinput'   % If there is input to the network, one might wish to find input-dependent fixed points 
            const_input = varargin{i+1};
        case 'tolfun'       % fminunc parameter - the tolerance on the objective function value for stopping
            tolfun = varargin{i+1};
        case 'tolx'         % fminunc parameter - the tolerance on how much the parameters move for the optimziation to stop
            tolx = varargin{i+1};
        case 'optmaxiters'  % maximum number of iterations for fminunc.
            opt_max_iters = varargin{i+1};            
        case 'display'      % Print information from fminunc for debugging or insight. ('off') or 'iter'.
            display = varargin{i+1};
        case 'epsfactor'    % How much to increase the randomness (epsilon) after a failure.
            eps_factor = varargin{i+1};	
        case 'dobail'       % If the optimization continues to fail the tolerances, should we exit or start over?
            do_bail = varargin{i+1};
        case 'dotopomap'    % Yes if interested in areas of slow speed.  This will return the x* of the first value less than fval_tol.
            do_topo_map = varargin{i+1};
        case 'stabilityanalysis'  % 'compact', 'none'.  Full means eigenvectors, compact means no eigenvectors, only eigenvalues.
            stability_analysis = varargin{i+1};
        otherwise
            assert ( false, 'Option not recognized.');
    end
end

% BP

N = net.layers(2).nPost;

% Memory allocation
fixed = cell(1,niters);
fvals = zeros(1,niters);
fp_starts = cell(1,niters);
evals = cell(1,niters);
evecs = cell(1,niters);
lvecs = cell(1,niters);
npos = zeros(1,niters);



optset = optimset('tolfun', tolfun,...	% termination based on function value (of the derivative)
    'hessian','on', ...
    'gradobj','on', ...
    'maxiter', opt_max_iters, ...
    'largescale', 'on', ...
    'display', display, ...
    'tolx', tolx ...  % if you want a really small fval, you need this to be smaller
    );
tic

nmeans = size(xmeans,2);

if ( niters < nmeans )
    mean_idxs = randi(nmeans, [1 niters]); % cover a random set of options
else
    mean_idxs = [[1:nmeans] randi(nmeans, [1 (niters-nmeans)])]; % cover all options, then random
end
%mean_idxs


parfor i = 1:niters
    myeps = epsilon;
    fixed{i} = zeros(N,1);
    fvals(i) = 1e10;
    midx = mean_idxs(i);
    while 1
        % Find a fixed point nearby.
        start_point = xmeans(:,midx) + myeps*randn(N,1);
        disp( ['Starting: ' num2str(i) ' with random point having norm: ' num2str(norm(start_point))  '.']);
        
        fp_starts{i} = start_point;
        [myfixed,fval, exitflag] = fminunc( @(x) find_one_fp(x, net, const_input, fval_tol, do_topo_map), start_point, optset);
        do_accept_opt_stop = false;
        switch exitflag
            case 1
                disp('Finished.  Magnitude of gradient smaller than the TolFun tolerance.');
                do_accept_opt_stop = true;
            case 2
                disp('Finished.  Change in x was smaller than the TolX tolerance.');
                do_accept_opt_stop = true;
            case 3 
                disp('Finished.  Change in the objective function value was less than the TolFun tolerance.');
                do_accept_opt_stop = true;
            case 5
                disp('Finished.  Predicted decrease in the objective function was less than the TolFun tolerance.');
                do_accept_opt_stop = true;
            case 0
                disp('Finished.  Number of iterations exceeded options.MaxIter or number of function evaluations exceeded options.FunEvals.');
                if do_topo_map
                    do_accept_opt_stop = true;  % Hard to see how the fval will be less than fval_tol here, but anyways...
                else
                    do_accept_opt_stop = false;
                end
            case -1
                disp('Finished.  Algorithm was terminated by the output function.');
                assert ( false, 'Still not sure what this case is.');
            otherwise
                assert ( false, 'New exit condition out of the fminunc optimizer front-end.');
        end
            
        assert ( isreal(myfixed), 'Not real!');        
        fixed{i} = myfixed;
        fvals(i) = fval;
        disp(fval)
        
        % Accept the fixed point if it's below the tolerance specified.  If one wants all local minima, then set dotopomap false and fval_tol high.
        if ( fval < fval_tol && do_accept_opt_stop )	            
            if strcmp(stability_analysis, 'full')
                with_eigenvectors = true;
                [ds, npos_i, Vs, Ls] = get_linear_stability_fp(net, fixed{i}, with_eigenvectors);
                evals{i} = ds;
                npos(i) = npos_i;
                evecs{i} = Vs;
                lvecs{i} = Ls;
            elseif strcmp(stability_analysis, 'compact')
                with_eigenvectors = false;
                [ds, npos_i, ~, ~] = get_linear_stability_fp(net, fixed{i}, with_eigenvectors);
                evals{i} = ds;
                npos(i) = npos_i;
                evecs{i} = [];
                lvecs{i} = [];
            else
                evals{i} = [];
                evecs{i} = [];
                lvecs{i} = [];
                npos(i) = 0;
            end
            disp( ['Finished ' num2str(i) ', ' num2str(npos(i)) ' positive eigenvalues for FP with norm: ' num2str(norm(fixed{i})), '.'] );
            myeps = epsilon;
            break;
        else
            myeps = myeps*eps_factor;
            if (myeps > max_eps)	% start over, no point in getting too big.
                if ( ~do_bail )
                    myeps = epsilon;
                else
                    fvals(i) = 1e30;
                    fixed{i} = NaN(N,1);
                    npos(i) = 1e30;
                    disp('Couldn''t find fixed point with desired tolerance. Bailing.');
                    break;
                end
            end
            if ~do_accept_opt_stop
                disp(['Try again for ' num2str(i) '. Point with norm ' num2str(norm(myfixed)) ' didn''t meet acceptable optimization termination criteria.']);
            else                
               disp(['Try again for ' num2str(i) '. Point with norm ' num2str(norm(myfixed)) ' didn''t meet function tolerance.']);
            end
            disp(['Epsilon sphere size: ' num2str(myeps)]);
        end
    end
end
toc

fpd = zeros(niters,niters);
for i = 1:niters
    for j = 1:niters
        fpd(i,j) = norm(fixed{i} - fixed{j});
    end
end


for i = 1:niters
    fp_struct(i).FP = fixed{i};
    fp_struct(i).FPNorm = norm(fixed{i});
    fp_struct(i).FPVal = fvals(i);
    fp_struct(i).nPos = npos(i);
    fp_struct(i).eigenValues = evals{i};
    fp_struct(i).eigenVectors = evecs{i};
    fp_struct(i).leftEigenVectors = lvecs{i};
    fp_struct(i).FPSearchStart = fp_starts(i);
end


disp('Fixed point search complete.');
%end


