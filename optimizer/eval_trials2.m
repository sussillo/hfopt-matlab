function varargout = eval_trials2(net, v_u_t, m_target_t, ~, v, lambda, pregen_forward_pass_t, TvV, trial_idxs, all_optional_params, all_simdata, ...
    do_return_network, do_return_J, do_return_J_grad, ...
    do_return_J_GaussNewton, do_return_preconditioner, do_return_J_and_network, do_return_J_grad_with_network, do_return_J_with_network, ...
    eval_network, eval_objfun, eval_gradient, eval_cg_afun, eval_preconditioner, eval_objfun_and_network, eval_gradient_with_network, eval_objfun_with_network, varargin)
% Written by David Sussillo (C) 2013
%
% This function evaluates all the forward passes / backward passes, etc.  It handles distributing the data and operations to the underlying
% subroutines, either parallel or serially, and then it coallates the results and returns them back to optimizer main routine.  As a result, there is
% a bit of a maze with respect to function interfaces.  There really isn't anything for it, but it does make for a pretty clean interface and an
% optimizer that can handle all kinds of networks and data.
%
%  The data could be a cell or an array. I support it.  The thing is that if it's an array, then underlying routine
% should support matrix operations over batches, whereas if it's cell (i.e. RNNs), then each trial for an RNN is quite
% the operation.  So  we divide things up differently depending on whether the data is a cell or a matrix.
%
%% NB NB NB
% Note that the subfunctions cannot know how to normalize correctly by the
% number of trials, so that happens here.  Ergo, don't normalize in
% any new subfunction you write.  It happens here.
%% NB NB NB

is_numeric_not_cell = isnumeric( v_u_t );

do_parallel = true;
do_wrappers = false;

optargin = size(varargin,2);
for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
        case 'doparallel'     % (true) Sometimes parallel is faster, sometimes it's slower
            do_parallel = varargin{i+1};
        case 'dowrappers'     % (false) Useful for gigantic datasets (in all_optional_params or net) so that all workers have a single copy.
            do_wrappers = varargin{i+1};
        otherwise
            assert ( false, ['No such option: ' varargin{i} '.']);
    end
end


all_evals = [do_return_network do_return_J do_return_J_grad do_return_J_GaussNewton do_return_preconditioner do_return_J_and_network do_return_J_grad_with_network do_return_J_with_network];
num_evals = sum(all_evals);
assert ( num_evals <= 2, 'Not doing it this way.  Using new functions for combos, since there is no need for generality.' );

if ~isempty(v_u_t)
    ntrials = size(v_u_t,2);
elseif ~isempty(m_target_t)
    ntrials = size(m_target_t,2);
end
assert ( ntrials > 0, 'stopped');


ncore_guess = matlabpool('size');
if ncore_guess == 0
    ncore_guess = 1;
end
% Determine the blocking for parallelism.  If matrix data, it's large
% blocks, if cell data, each block is a trial.
if is_numeric_not_cell
    nblocks = ncore_guess;  % Something reasonable based on the matlab pool size.
else
    nblocks = ntrials;
end

% This is because the parfor is complaining loudly when these come in empty, even though they are protected by the
% conditionals and thus won't be used.
if isempty(v_u_t)
    if is_numeric_not_cell
        v_u_t = [];
    else
        v_u_t = cell(1,nblocks);
    end
end
if isempty(m_target_t)
    if is_numeric_not_cell
        m_target_t = [];
    else
        m_target_t = cell(1,nblocks);
    end
end
if isempty(pregen_forward_pass_t)
    if is_numeric_not_cell
        pregen_forward_pass_t = [];
    else
        pregen_forward_pass_t = cell(1,nblocks);
    end
end

TRAINING = 1;
VALIDATION = 2;

% Shouldn't force the laborious crap on the end user. DCS:11/22/2011
if length(trial_idxs) < ntrials
    trial_idxs = ones(1,ntrials);
end


% Now the blocks are collected in cell arrays.  Everything but the forward
% pass is averaged and returned as a scalar or a vector.  The forward pass
% is put back together as a matrix if that's what it was to begin with.
new_simdata_blocks = {};
forward_pass_blocks = {};
objfun = {};
all_objfuns = {};
grad = {};
gv = {};

chunk_size = ceil(ntrials/nblocks);

% Optional params is 2 cells (training and validation) of a structure
% array, where each structure is one trial, note these index either the full training set or the full test set, not just a subset.
if ~isempty ( all_optional_params )
    if TvV == TRAINING
        all_optional_params = all_optional_params{TRAINING};
    else
        all_optional_params = all_optional_params{VALIDATION};
    end
end

% all_simdata is 2 cells (training and validation) containing a structure array, where each structure is one trial, note
% these index either the full training set or the full test set, not just a subset.
all_simdata1 = {};
if ~isempty ( all_simdata )
    if TvV == TRAINING
        all_simdata1 = all_simdata{TRAINING};
    else
        all_simdata1 = all_simdata{VALIDATION};
    end
end

%%% Note that trials aren't really well defined for block data.  One easy hack is to let the eval_trials system build out T all_simdata, and then
%%% block that up into nblocks.  I think it'll work for my one and only application of eval_trials2.m to numeric data, but it's short-sighted.  The
%%% right thing to do is have nblock simdata, for numeric data, I think.  But that seems funny, also.

block_idxs = cell(1,nblocks);
block_idxs{1} = 1:chunk_size;
for i = 2:nblocks
    block_idxs{i} = block_idxs{i-1}(end)+1:block_idxs{i-1}(end)+chunk_size;
end
block_idxs{end}(block_idxs{end} > ntrials) = [];

block_trial_idxs = cell(1,nblocks);
block_all_optional_params = cell(1,nblocks);  % Each cell contains a structure array with the right trials.
block_all_simdata = cell(1,nblocks);
for i = 1:nblocks
    block_trial_idxs{i} = trial_idxs(block_idxs{i});
    if ~isempty(all_optional_params)
        block_all_optional_params{i} = all_optional_params(block_trial_idxs{i});
    else
        empty_all_optional_params(length(block_trial_idxs{i})).nothing = '';  %#ok<AGROW> % Initing last element of struct array for empty array.
        block_all_optional_params{i} = empty_all_optional_params;
    end
    if ~isempty(all_simdata1)
        block_all_simdata{i} = all_simdata1(block_trial_idxs{i});
    else
        empty_all_simdata(length(block_trial_idxs{i})).nothing = '';  %#ok<AGROW> % Initing last element of struct array for empty array.
        block_all_simdata{i} = empty_all_simdata;
    end
end

v_u_t_block = cell(1,nblocks);
m_target_t_block = cell(1,nblocks);
if is_numeric_not_cell && do_parallel && ~do_wrappers
    for i = 1:nblocks
        v_u_t_block{i} = v_u_t(:,block_idxs{i});
        m_target_t_block{i} = m_target_t(:,block_idxs{i});
    end
end

pregen_forward_pass_t_blocks = cell(1,nblocks);
if (do_return_J_GaussNewton || do_return_J_grad_with_network || do_return_J_with_network) && is_numeric_not_cell
    %assert ( false, 'Verify this makes sense.');  % DCS 5/6/2013
    nreturns = length(pregen_forward_pass_t); %  Number of return items in forward pass, e.g. I->H1->H2->O is 4
    for i = 1:nblocks
        for j = 1:nreturns
            pregen_forward_pass_t_blocks{i}{j} = pregen_forward_pass_t{j}(:,block_idxs{i});
        end
    end
end




if ~do_parallel
    
    if do_return_network
        forward_pass_blocks = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        for i = 1:nblocks
            if is_numeric_not_cell
                package = eval_network(net, v_u_t(:,block_idxs{i}), m_target_t(:,block_idxs{i}), TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            else
                package = eval_network(net, v_u_t{i}, m_target_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            end
            
            forward_pass_blocks{i} = package{1};
            new_simdata_blocks{i} = package{end};
        end
    end
    if do_return_J
        objfun = cell(1,nblocks);
        all_objfuns = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        for i = 1:nblocks
            if is_numeric_not_cell
                package = eval_objfun(net, v_u_t(:,block_idxs{i}), m_target_t(:,block_idxs{i}), TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            else
                package = eval_objfun(net, v_u_t{i}, m_target_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            end
            objfun{i} = package{1};
            all_objfuns{i} = package{2};
            new_simdata_blocks{i} = package{end};
        end
    end
    if do_return_J_with_network
        objfun = cell(1,nblocks);
        all_objfuns = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        for i = 1:nblocks
            if is_numeric_not_cell
                package = eval_objfun_with_network(net, v_u_t(:,block_idxs{i}), m_target_t(:,block_idxs{i}), pregen_forward_pass_t_blocks{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            else
                package = eval_objfun_with_network(net, v_u_t{i}, m_target_t{i}, pregen_forward_pass_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            end
            objfun{i} = package{1};
            all_objfuns{i} = package{2};
            new_simdata_blocks{i} = package{end};
        end
    end
    if do_return_J_grad
        %grad = cell(1,nblocks);
        avg_grad = zeros(size(net.theta));
        new_simdata_blocks = cell(1,nblocks);
        for i = 1:nblocks
            if is_numeric_not_cell
                package = eval_gradient(net, v_u_t(:,block_idxs{i}), m_target_t(:,block_idxs{i}), TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            else
                package = eval_gradient(net, v_u_t{i}, m_target_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            end
            %grad{i} = package{1};
            avg_grad = avg_grad + package{1};
            new_simdata_blocks{i} = package{end};
        end
        avg_grad = avg_grad / ntrials;
    end
    if do_return_J_GaussNewton
        %gv = cell(1,nblocks);
        avg_gv = zeros(size(net.theta));
        new_simdata_blocks = cell(1,nblocks);
        for i = 1:nblocks
            if is_numeric_not_cell
                
                % v and forward_pass_t are parameters to eval_trials.m,
                % note that the forward_pass_t is already broken up
                % appropriately.
                package = eval_cg_afun(net, v_u_t(:,block_idxs{i}), m_target_t(:,block_idxs{i}), v, lambda, pregen_forward_pass_t_blocks{i}, TvV, ...
                    block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                
            else
                % v and forward_pass_t are parameters to eval_trials.m
                package = eval_cg_afun(net, v_u_t{i}, m_target_t{i}, v, lambda, pregen_forward_pass_t{i}, TvV, ...
                    block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            end
            %gv{i} = package{1};
            avg_gv = avg_gv + package{1};
            new_simdata_blocks{i} = package{end};
        end
        avg_gv = avg_gv / ntrials;
    end
    if do_return_preconditioner
        precon = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        for i = 1:nblocks
            if is_numeric_not_cell
                package = eval_preconditioner(net, v_u_t(:,block_idxs{i}), m_target_t(:,block_idxs{i}), lambda, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            else
                package = eval_preconditioner(net, v_u_t{i}, m_target_t{i}, lambda, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            end
            precon{i} = package{1};
            new_simdata_blocks{i} = package{end};
        end
    end
    
    if do_return_J_and_network
        forward_pass_blocks = cell(1,nblocks);
        objfun = cell(1,nblocks);
        all_objfuns = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        for i = 1:nblocks
            if is_numeric_not_cell
                package = eval_objfun_and_network(net, v_u_t(:,block_idxs{i}), m_target_t(:,block_idxs{i}), TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            else
                package = eval_objfun_and_network(net, v_u_t{i}, m_target_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            end
            forward_pass_blocks{i} = package{1};  % The forward pass goes first.
            objfun{i} = package{2};  % Objective function second.
            all_objfuns{i} = package{3};
            new_simdata_blocks{i} = package{end};
        end
    end
    
    if do_return_J_grad_with_network
        avg_grad = zeros(size(net.theta));
        %grad = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        for i = 1:nblocks
            if is_numeric_not_cell
                package = eval_gradient_with_network(net, v_u_t(:,block_idxs{i}), m_target_t(:,block_idxs{i}), pregen_forward_pass_t_blocks{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            else
                package = eval_gradient_with_network(net, v_u_t{i}, m_target_t{i}, pregen_forward_pass_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
            end
            %grad{i} = package{1};
            avg_grad = avg_grad + package{1};
            new_simdata_blocks{i} = package{end};
        end
        avg_grad = avg_grad / ntrials;
    end
    
end


if do_parallel && ~do_wrappers
    
    if do_return_network
        forward_pass_blocks = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        if is_numeric_not_cell
            parfor i = 1:nblocks
                package = feval(eval_network, net, v_u_t_block{i}, m_target_t_block{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                forward_pass_blocks{i} = package{1};
                new_simdata_blocks{i} = package{end};
            end
        else
            parfor i = 1:nblocks
                package = feval(eval_network, net, v_u_t{i}, m_target_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                forward_pass_blocks{i} = package{1};
                new_simdata_blocks{i} = package{end};
            end
        end
    end
    if do_return_J
        objfun = cell(1,nblocks);
        all_objfuns = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        if is_numeric_not_cell
            parfor i = 1:nblocks
                package = feval(eval_objfun, net, v_u_t_block{i}, m_target_t_block{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                objfun{i} = package{1};
                all_objfuns{i} = package{2};
                new_simdata_blocks{i} = package{end};
            end
        else
            parfor i = 1:nblocks
                package = feval(eval_objfun, net, v_u_t{i}, m_target_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                objfun{i} = package{1};
                all_objfuns{i} = package{2};
                new_simdata_blocks{i} = package{end};
            end
        end
    end
    if do_return_J_with_network
        objfun = cell(1,nblocks);
        all_objfuns = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        if is_numeric_not_cell
            parfor i = 1:nblocks
                package = feval(eval_objfun_with_network, net, v_u_t_block{i}, m_target_t_block{i}, pregen_forward_pass_t_blocks{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                objfun{i} = package{1};
                all_objfuns{i} = package{2};
                new_simdata_blocks{i} = package{end};
            end
        else
            parfor i = 1:nblocks
                package = feval(eval_objfun_with_network, net, v_u_t{i}, m_target_t{i}, pregen_forward_pass_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                objfun{i} = package{1};
                all_objfuns{i} = package{2};
                new_simdata_blocks{i} = package{end};
            end
        end
    end
    if do_return_J_grad
        avg_grad = zeros(size(net.theta));
        %grad = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        if is_numeric_not_cell
            parfor i = 1:nblocks
                package = feval(eval_gradient, net, v_u_t_block{i}, m_target_t_block{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                %grad{i} = package{1};
                avg_grad = avg_grad + package{1};
                new_simdata_blocks{i} = package{end};
            end
        else
            parfor i = 1:nblocks
                package = feval(eval_gradient, net, v_u_t{i}, m_target_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                %grad{i} = package{1};
                avg_grad = avg_grad + package{1};
                new_simdata_blocks{i} = package{end};
            end
        end
        avg_grad = avg_grad / ntrials;
    end
    if do_return_J_GaussNewton
        %gv = cell(1,nblocks);
        avg_gv = zeros(size(net.theta));
        new_simdata_blocks = cell(1,nblocks);
        if ( is_numeric_not_cell )
            parfor i = 1:nblocks
                % v and forward_pass_t are parameters to eval_trials.m,
                % note that the forward_pass_t is already broken up
                % appropriately.
                package = feval(eval_cg_afun, net, v_u_t_block{i}, m_target_t_block{i}, v, lambda, pregen_forward_pass_t_blocks{i}, TvV, ...
                    block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                %gv{i} = package{1};
                avg_gv = avg_gv + package{1};
                new_simdata_blocks{i} = package{end};
            end
        else
            parfor i = 1:nblocks
                % v and forward_pass_t are parameters to eval_trials.m
                package = feval(eval_cg_afun, net, v_u_t{i}, m_target_t{i}, v, lambda, pregen_forward_pass_t{i}, TvV, ...
                    block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                %gv{i} = package{1};
                avg_gv = avg_gv + package{1};
                new_simdata_blocks{i} = package{end};
            end
        end
        avg_gv = avg_gv / ntrials;
    end
    if do_return_preconditioner
        precon = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        if is_numeric_not_cell
            parfor i = 1:nblocks
                package = feval(eval_preconditioner, net, v_u_t_block{i}, m_target_t_block{i}, lambda, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                precon{i} = package{1};
                new_simdata_blocks{i} = package{end};
            end
        else
            parfor i = 1:nblocks
                package = feval(eval_preconditioner, net, v_u_t{i}, m_target_t{i}, lambda, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                precon{i} = package{1};
                new_simdata_blocks{i} = package{end};
            end
        end
    end
    
    if do_return_J_and_network
        forward_pass_blocks = cell(1,nblocks);
        objfun = cell(1,nblocks);
        all_objfuns = cell(1,nblocks);
        new_simdata_blocks = cell(1,nblocks);
        if is_numeric_not_cell
            parfor i = 1:nblocks
                package = feval(eval_objfun_and_network, net, v_u_t_block{i}, m_target_t_block{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                forward_pass_blocks{i} = package{1};  % The forward pass goes first.
                objfun{i} = package{2};  % Objective function second.
                all_objfuns{i} = package{3};
                new_simdata_blocks{i} = package{end};
            end
        else
            parfor i = 1:nblocks
                package = feval(eval_objfun_and_network, net, v_u_t{i}, m_target_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                forward_pass_blocks{i} = package{1};  % The forward pass goes first.
                objfun{i} = package{2};  % Objective function second.
                all_objfuns{i} = package{3};
                new_simdata_blocks{i} = package{end};
            end
        end
    end
    if do_return_J_grad_with_network
        %grad = cell(1,nblocks);
        avg_grad = zeros(size(net.theta));
        new_simdata_blocks = cell(1,nblocks);
        if is_numeric_not_cell
            parfor i = 1:nblocks
                package = feval(eval_gradient_with_network, net, v_u_t_block{i}, m_target_t_block{i}, pregen_forward_pass_t_blocks{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                %grad{i} = package{1};
                avg_grad = avg_grad + package{1};
                new_simdata_blocks{i} = package{end};
            end
        else
            
            parfor i = 1:nblocks
                package = feval(eval_gradient_with_network, net, v_u_t{i}, m_target_t{i}, pregen_forward_pass_t{i}, TvV, block_trial_idxs{i}, block_all_optional_params{i}, block_all_simdata{i});
                %grad{i} = package{1};
                avg_grad = avg_grad + package{1};
                new_simdata_blocks{i} = package{end};
            end            
        end
        avg_grad = avg_grad / ntrials;
    end
    
end

if ~isempty(all_simdata)
    new_simdata_tv = all_simdata{TvV};
    %disp(num2str(length(new_simdata_tv)))
    new_simdata_tv(trial_idxs) = [new_simdata_blocks{:}];
    %disp(num2str(length(new_simdata_tv)))
    all_simdata{TvV} = new_simdata_tv;
    %disp(num2str(trial_idxs))
end



% I'm really not sure if this code helps or if I've even implemented it correctly, so I'm pulling it out for now.
if do_parallel && do_wrappers
    assert ( false, 'You lazy stopp.');  % I'm talking to myself here.
end

if do_return_J || do_return_J_and_network || do_return_J_with_network
    J = 0;
    avg_all_Js = zeros(size(all_objfuns{1}));
end
if 0 && (do_return_J_grad || do_return_J_grad_with_network)
    avg_grad = zeros(size(grad{1}));
end
if 0 && do_return_J_GaussNewton
    avg_gv = zeros(size(gv{1}));
end
if do_return_preconditioner
    avg_precon = zeros(size(precon{1}));
end


% At least doubles memory here, but not sure what else to do.
if is_numeric_not_cell && ~isempty(forward_pass_blocks)
    nreturns = length(forward_pass_blocks{1});
    fp_array_of_cells = reshape([forward_pass_blocks{:}], nreturns, nblocks);
    forward_pass = cell(1,nreturns);
    for i = 1:nreturns
        forward_pass{i} = cell2mat(fp_array_of_cells(i,:));
    end
else
    forward_pass = forward_pass_blocks;
end

% Have to do the same thing for the double return case because the anonymous functions won't return two things.
for i = 1:nblocks
    if do_return_J || do_return_J_and_network || do_return_J_with_network
        J = J + objfun{i};
        avg_all_Js = avg_all_Js + all_objfuns{i};
    end
    if 0 && (do_return_J_grad || do_return_J_grad_with_network)
       avg_grad = avg_grad + grad{i};
    end
    if 0 && do_return_J_GaussNewton
       avg_gv = avg_gv + gv{i};
    end
    if do_return_preconditioner
        avg_precon = avg_precon + precon{i};
    end
end

if isfield(net, 'modMasksByTrial')
    ntrials_mod_masks_by_trial = sum(net.modMasksByTrial,2); % 6 x ntrials
    ntrials_Wru_calc = ntrials_mod_masks_by_trial(1);
    ntrials_Wrr_calc = ntrials_mod_masks_by_trial(2);
    ntrials_Wzr_calc = ntrials_mod_masks_by_trial(3);
    %ntrials_x0_calc = ntrials_mod_masks_by_trial(4);  % See below
    ntrials_bx_calc = ntrials_mod_masks_by_trial(5);
    ntrials_bz_calc = ntrials_mod_masks_by_trial(6);
else
    ntrials_Wru_calc = ntrials;
    ntrials_Wrr_calc = ntrials;
    ntrials_Wzr_calc = ntrials;
    %ntrials_x0_calc = ntrials;   % See below
    ntrials_bx_calc = ntrials;
    ntrials_bz_calc = ntrials;
end

% What we really need is a number, ntrials_per_ic, and this is what we divide by.
% If there is a single IC for all trials, then dividing by ntrials
% is appropriate.  Otherwise, we'll divide too much and the IC won't be learned properly.
% Note also, that this means there was probably an IC bug before where ics were divided out by too large a number,
% if there were multiple learnable ics.
if isfield(net, 'NTrialsPerIC')
    ntrials_x0_calc = net.NTrialsPerIC;  % This is a row vector with size equal to the number of ICs.
else
    ntrials_x0_calc = ntrials;  % This is broken except for one IC for all trials, put preserves backwards compatability.
end
assert ( ntrials_Wru_calc > 0, 'stopped');
assert ( ntrials_Wrr_calc > 0, 'stopped');
assert ( ntrials_Wzr_calc > 0, 'stopped');
assert ( isempty(find(ntrials_x0_calc == 0,1)), 'stopped');
assert ( ntrials_bx_calc > 0, 'stopped');
assert ( ntrials_bz_calc > 0, 'stopped');

% Note that the means the sub functions no longer divide out the block # of
% trials.
if do_return_J || do_return_J_and_network || do_return_J_with_network
    % I think the error has to be divided out by all the trials, regardless of what is modified.
    % If this needs to be separated into training and validation, sets, it'd be a plot function.
    % Another way to think about it, is that there is an error you are trying to minimize, and the
    % various modification masks over structure and trials are various ways of reducing that error.
    J = J / ntrials;
    avg_all_Js = avg_all_Js / ntrials;
end
% Note that adding this code breaks the abstraction, and now eval_trials2.m assumes the network will
% have this particular form.  I'm not sure what else to do.  Not going to solve it now. DCS:2/1/2014
% In the end, we'll have to define another function that the various networks conform to, to do this averaging.
% Then eval_trials2.m will call that function. DCS:2/1/2014
if 0 && (do_return_J_grad || do_return_J_grad_with_network)
    if 0
        [n_Wruag_v, n_Wrrag_n, m_Wzrag_n, n_ICx0ag_c, n_bxag_1, m_bzag_1] = unpackRNN(net, avg_grad);
        if 0  % This should work, but it doesn't.
            n_Wruag_v = n_Wruag_v / ntrials_Wru_calc;
            n_Wrrag_n = n_Wrrag_n / ntrials_Wrr_calc;
            m_Wzrag_n = m_Wzrag_n / ntrials_Wzr_calc;
            N = size(n_Wruag_v,1);
            n_ICx0ag_c = n_ICx0ag_c ./ repmat(ntrials_x0_calc, N, 1);
            n_bxag_1 = n_bxag_1 / ntrials_bx_calc;
            m_bzag_1 = m_bzag_1 / ntrials_bz_calc;
        else
            n_Wruag_v = n_Wruag_v / ntrials;
            n_Wrrag_n = n_Wrrag_n / ntrials;
            m_Wzrag_n = m_Wzrag_n / ntrials;
            N = size(n_Wruag_v,1);
            n_ICx0ag_c = n_ICx0ag_c / ntrials;
            n_bxag_1 = n_bxag_1 / ntrials;
            m_bzag_1 = m_bzag_1 / ntrials;
        end
        avg_grad = packRNN(net, n_Wruag_v, n_Wrrag_n, m_Wzrag_n, n_ICx0ag_c, n_bxag_1, m_bzag_1);
    end
    avg_grad = avg_grad / ntrials;
end
if 0 && do_return_J_GaussNewton
    if 0
        if 0  % This should work, but it doesn't.
            [n_Wrugv_v, n_Wrrgv_n, m_Wzrgv_n, n_ICx0gv_c, n_bxgv_1, m_bzgv_1] = unpackRNN(net, avg_gv);
            n_Wrugv_v = n_Wrugv_v / ntrials_Wru_calc;
            n_Wrrgv_n = n_Wrrgv_n / ntrials_Wrr_calc;
            m_Wzrgv_n = m_Wzrgv_n / ntrials_Wzr_calc;
            N = size(n_Wrugv_v,1);
            n_ICx0gv_c = n_ICx0gv_c ./ repmat(ntrials_x0_calc, N, 1);
            n_bxgv_1 = n_bxgv_1 / ntrials_bx_calc;
            m_bzgv_1 = m_bzgv_1 / ntrials_bz_calc;
        else
            [n_Wrugv_v, n_Wrrgv_n, m_Wzrgv_n, n_ICx0gv_c, n_bxgv_1, m_bzgv_1] = unpackRNN(net, avg_gv);
            n_Wrugv_v = n_Wrugv_v / ntrials;
            n_Wrrgv_n = n_Wrrgv_n / ntrials;
            m_Wzrgv_n = m_Wzrgv_n / ntrials;
            N = size(n_Wrugv_v,1);
            n_ICx0gv_c = n_ICx0gv_c / ntrials;
            n_bxgv_1 = n_bxgv_1 / ntrials;
            m_bzgv_1 = m_bzgv_1 / ntrials;
        end
        avg_gv = packRNN(net, n_Wrugv_v, n_Wrrgv_n, m_Wzrgv_n, n_ICx0gv_c, n_bxgv_1, m_bzgv_1);
    end
    avg_gv = avg_gv / ntrials;
end
if do_return_preconditioner
    assert ( false, 'stopped'); % Adding the initial condition code breaks eval_trials for feed forward networks.
    avg_precon = avg_precon / nblocks;  % Note nblocks!!! Not ntrials!!!
end

% The anonymous functions force one to return only a single output, so things get wrapped as necessary and varargout only has one argument.
varargout = {};
if do_return_network || do_return_J_and_network
    varargout{end+1} = forward_pass;
end
if do_return_J || do_return_J_and_network || do_return_J_with_network
    varargout{end+1} = J;
    varargout{end+1} = avg_all_Js;
end
if do_return_J_grad || do_return_J_grad_with_network
    varargout{end+1} = avg_grad;
end
if do_return_J_GaussNewton
    varargout{end+1} = avg_gv;
end
if do_return_preconditioner
    varargout{end+1} = avg_precon;
end


varargout{end+1} = all_simdata;  % Always last.
varargout = {varargout};

