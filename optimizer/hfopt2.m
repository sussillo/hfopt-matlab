function [theta, objfun_train, objfun_test, stats, all_simdata] = hfopt2(net, v_inputtrain_T_start, m_targettrain_T_start, v_inputtest_t_start, m_targettest_t_start, varargin)
% function [theta, objfun_train, objfun_test, stats] = hfopt2(net, v_inputtrain_T, m_targettrain_T, v_inputtest_t, m_targettest_t, varargin)
%
% Written by David Sussillo (C) 2013
%
% This is verison 2.0 of my generic implementation of the HF optimizer as put forward by Martens in his ICML 2010
% paper.  This version of the optimizer takes either simple data, or it takes functions that will generate the data
% at each pass.  This is done very carefully so that gradient computation, and the Hessian-vector multiplication
% computation happen with the exact same data.  The following explanation is much simpler if you are simply passing in
% data, but if you are passing in functions that generate the data, then what happens on each HF iteration is as follows:
% 
% 1)  The inputs are generated.  
% 2)  Some optional arguments are evaluated (which influence the run)
% 3)  The forward pass of the network is evaluated (there is one -and only one- forward pass.
% 4)  The the targets are generated (potentially based on the output of the objective function evaluation (simdata) and the
%     previous inputs).
% 5)  The objective function is evaluated from the forward pass.
% 6)  The input / forward_pass / targets are (possibly) downsampled to a minibatch for the CG iteration.
% 7)  The gradient is computed on the full dataset.  
% 8)  The conjugate gradient algorithm is used to compute the Hessian vector product.
% 9)  CG backtracking is used to determine which of the CG steps had the lowest error.  Each backtracking step
%     requires a full forward pass.
% 10) If the CG backtracking found a better solution (than this iterations forward pass), that solution is taken and
%     a very simple backwards line search is optionally done.
% 11) A whole mess of cases are examined and determine whether to continue the optimization, save the data, etc.

% DATA REPRESENTATION
% This routine supports both a cell input / target type, as well as a matrix.  The matrix representation only comes
% in deep networks, whereas the RNN stuff exclusively uses the cell data representation.  It's been awhile since I've
% played around with deep nets, so if you implement deep nets, you should make sure things are working as you expect
% them to.  
% 
% Support an array of data, or a cell array of data, or a function handle that returns a two cell array of data (input /
% target).  A function handle implicitly implies that the we'll resample from the function at each HF step, else why
% not just pass the data?
%
% If you pass a cell array, then the assumption is that each element of the cell array is a trial.  For example, the
% RNN, {{input1} {input2}}, this has two trials, and the input should have size VxT, where V is input dimension and T
% is the number of time points.
%
% One can sample from the data in two ways for the minibatches.  The data is: examples as rows in either cells or a
% matrix.  The training dataset can be sampled either with random rows or random contiguous blocks: sample_style -
% 'random_rows' or 'random_blocks'.
%
%  Random rows simply picks random elments (which could be matrices of temporral data, for example, if using a cell array)
%  from the T sized data set and uses those.  Random blocks picks S continguous random indices from the T sized training
%  dataset (which again could be whole matrices of data if using a cell array) and uses those.  I'm pretty sure that
%  random rows really is the same as 'random_columns', because I always supply everything as a cell array with a row
%  1xN layout.  Sorry for the name-o.

% Common variable names that you will see in this file, and that you may have to work with.

% net - the this the network structure that holds the parameters
% simparams - this is a structure of parameters defined by the user and used by other used defined functions, such as the functions which define
%    the inputs ands and the targets. 
% funs - this is the set of functions for calling the forward pass, gradient, objective function, etc.  
% cg_found_better_solution - did the CG backtracking find a better solution? yes / no
% f2 - ?
% random_trial_idxs - These are the random trial indices used for this minibatch,
% forward_pass_T - a cell array or matrix (# trials) holding the forward pass activiations, etc.  This structure is different for different 
%    types of networks that are optimized.  For the recurrent network, it looks like
%
% forward_pass_s - same as just above, except for the minibatch used in the CG iterations           
% v_inputtrain_T - cell array (or matrix) (# trials) used for the inputs.  For the RNN, each element of the cell array is a VxT array of inputs
% v_inputtrain_s - same as just above, except for the minibatch used in the CG iterations,
% v_inputtest_t - same as just above, except used for the validation set,
% m_targettrain_T - cell array (or matrix) (# trials) used for the targets.  For the RNN, each element of the cell array is a MxT array of targets
% m_targettrain_s - same as just above, except for the minibatch used in the CG iterations, 
% m_targettest_t - same as just above, except for the validation set,
%
% User defined structures.
% all_optional_args - these are user defined parameters, which are set at the beginning of each HF iteration (see order above) by calling a user
%     defined function
% all_simdata - these are user defined data, which are set by the input / target / forward pass functions.  In this case, you'll want to use these to
%     go from inputs, which are called in a separate function call from the function used to generate the targets.  This data structure is a bridge.
%  
% Both of the above variables have the same structure. Its for is all_optional_args{TvV_T}(trial_id) is a user defined structure, 
% where TvV_T = 1 or 2, for either training or test, respectively, and trial_id is the gives the trial.  Thus there is a cell array for 
% training vs. validation, and inside is a structure array.  An example of a reasonble eval fun parameter to supply is a random noise seed for each trial.
%
% The difference between all_simdata and all_optional_args is that simdata is controlled by the functions that create the inputs and
% targets, and also controlled by the forward pass function.  The all_optional_args is called once per iteration and set by a user defined
% function.
            

%% VARIABLE INITIALIZATIONS 
% See the optional arguments section for details about all the inputs

% The default assumption is that all of the various evaluation subroutines should run serially, and that the object
% wrapper script (WOW) will not be invoked.  The wrapper script is useful for very large datasets, in order to make the
% parfor more efficient.  If there is no large dataset, things will become much slower, presumably due to overhead. 
% Currently the wrappers are disabled, so you should never set them to true.  -DCS:2013/11/19
do_parallel_network = false;
do_parallel_objfun = false;
do_parallel_gradient = false;
do_parallel_cg_afun = false;
do_parallel_precon = false;

do_wrappers_network = false;
do_wrappers_objfun = false;
do_wrappers_gradient = false;
do_wrappers_cg_afun = false;
do_wrappers_precon = false;

% For investigating the nature of the error function.  Be careful with size!
do_save_thetas = false; 

all_optional_args = {};			% for eval functions
%all_simdata = {};   % For all the eval functiosn, a generic cell array, the matlab equivalent of a void* being passed around.

sample_style = 'random_rows'; % random_blocks

best_test_net = net;  % This is used in the case that people care about validation error.
best_test_net_hf_iter = 1;
best_test_net_objfun_train = Inf;
best_test_net_objfun_test = Inf;

save_path = './';
save_every = 10;			% save every 'save_every' hf iterations.
filename_part = '';

do_plot = 1;
do_plot_all_objectives = true;
display_level = Inf;

% Leaving a few decimal points here for shennanigans.
objfun_min = 0.0;           % Occasionally, one knows what a good objfun value is.
objfun_tol = 1e-29;			% matter a lot for grinding at the end
tolfun = 1e-29;             % Stopping condition for the magnitude of the gradient, following fminunc.
cgtol = 1e-29;
max_lambda = 1e29;           % A stopping condition.

net_type = 'rnn-trials';
cgbt_objfun = 'train';			% sthopping criterion is based on training or testing data (testing helps
weight_cost = 0.0;			% matters a lot

% with grinding)

% I've deciced the default should be to never stop, unless the user specifies it with the following conditions.
max_hf_iters = 5000;			% maximum number of HF iterations[
hf_max_bork_count = 5000;                 % maximum times TOTAL that HF fails to find better solution before we quit.
hf_max_consecutive_bork_count = 5000;	% maximum times CONSECUTIVELY that HF fails before we quit.  this should be related to how quickly lambda changes
hf_max_consecutive_test_iter_increases = Inf;  % Used to stop if the number of consecutive HF iters with higher validation error reaches a value.
total_consecutive_test_increase_count = 0;
min_test_objfun = Inf;

lowest_maxcg = 20;
highest_maxcg = 300;			% this number definitely matters, otherwise the optimization flattens out
% early.  does it help though to grow to any amount, or just cg grinding?
lowest_lambda = realmin;			% lambda can become stupidly small and that's not helpful

gamma = 1.3;
cg_increase_factor = 1.3; % if we find a solution with CG, then let CG run as many iterations as this time multiplied
% by this factor.

init_lambda = 1.0;
do_grad_on_full_data = 1;
last_w_decay_factor = 0.95; % found this in Martens implementation, so theta for init in CG is decay * theta

% These are the values described in the paper.
rho_drop_thresh = 0.75;   % the value at which lambda is reduced
rho_boost_thresh = 0.25;  % the value at which lambda is increased
rho_drop_val = 2.0/3.0;
rho_boost_val = 1.0/rho_drop_val;

do_time_hf_iters = 1;

max_lambda_increases_for_negative_curvature = 500;
smallest_pAp = Inf;

Suser = NaN;
Sfrac_user = NaN;

do_line_search = true;
do_user_defined_learning_rate = false;
user_defined_learning_rate = 1.0;
user_defined_learning_rate_decay = 1.0;

optional_eval_fun = [];
optional_plot_fun = [];

did_change_train_data = false;
did_change_test_data = false;
simparams = [];  % Stuff to save for the user, just so it's tied to the end result of the simulation.  This would usually be things like the parameters to recreate the inputs, etc.

% Save the name of the file and the file contents itself, if a user so
% desires.
train_file_name = [];
train_file_contents = [];

nhfiters_to_reset_matlabpool = Inf;

all_simdata = [];

optargin = size(varargin,2);
for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
        case 'nhfiterstoresetmatlabpool'
            nhfiters_to_reset_matlabpool = varargin{i+1};
        case 'rhodropthresh'		
            % the value of rho at which lambda decreases
            rho_drop_thresh = varargin{i+1};
        case 'rhoboostthresh'
            % the value of rho at which lambda increases
            rho_boost_thresh = varargin{i+1};
        case 'rhodropval'
            % The multiplier by which lambda is decreased
            rho_drop_val = varargin{i+1};
        case 'rhoboostval'
            % The multiplier by which lambda is increased
            rho_boost_val = varargin{i+1};
        case 'doplot'
            % Should the optimizer call the supplied plot function?  The plot function is extremely useful for debugging
            % and insuring that you actually are optimizing what you think you are optimizing.
            do_plot = varargin{i+1};
        case 'doplotallobjectives'
            do_plot_all_objectives = varargin{i+1};
        case 'displaylevel'
            % How much optimizer output do you want to see?  I've roughly prioritized things so that a level of 0 gives nothing and 3 gives
            % everything.
            display_level = varargin{i+1};
        case 'nettype'
            % The optimizer handles different types of networks.  I'm
            % currently only releasing it with 'rnn-trials' code.  Turned
            % off for now.
            net_type = varargin{i+1};
            %assert ( strcmpi(net_type, 'rnn-trials'), 'Stopped');            
        case 'objfunmin'
            % The objective function value at which to stop the optimization.  Initialized to 0.0
            objfun_min = varargin{i+1};
        case 'objfuntol'
            % The absolute tolerance of the objective function. If the update is less than this value, the optimization is stopped.  Set to 1e-29.
            objfun_tol = varargin{i+1};
        case 'tolfun'
            % The tolerance on the norm of the gradient.  If the norm of the gradient is smaller than this value, the optimization is stopped. Set to
            % essentially 1e-29.
            tolfun = varargin{i+1};
        case 'maxhfiters'
            % The maximum number of HF iterations (HF iterations are the outer loop iteration.  Default = 5000
            max_hf_iters = varargin{i+1};
        case 'maxlambda'
            % The maximum lambda value before the optimization stops.  Default = 1e29.
            max_lambda = varargin{i+1};
        case {'maxhffailures', 'maxcgfailures'}
            % The maximum number of times that the optimizer fails to find an improvement in the objective function before quitting.  Default = 5000.
            hf_max_bork_count = varargin{i+1};            
        case 'maxconsecutivecgfailures'
            % The maximum number of consecutive times that the optimizer fails to find an improvement in the objective function before quitting.
            % Default = 5000.
            hf_max_consecutive_bork_count = varargin{i+1};
        case 'maxconsecutivetestincreases'
            % Used to stop if the number of consecutive HF iters with higher validation error reaches a value.  Default = Inf
            hf_max_consecutive_test_iter_increases = varargin{i+1};
        case 'weightcost'
            % The L2 weight cost is a parameter that somehow early on was not tied to the model.  In hindsight this was a mistake.  Basically, my
            % thinking was that I wanted an easy way to evaluate the objective function without the weight cost involved.  Anyways, this is where you
            % set the weight cost for the parameters.
            weight_cost = varargin{i+1};
        case 'initlambda'
            % The initial lambda value.  This value really matters for getting the optimization started right.  The heuristic is to set it as low as
            % possible without having the optimization fail all the time.  If you set it too high, it could take a while for the lambda heuristics to
            % figure out that the lambda value should be lower, in which case you've wasted a lot of compute time.  Default = 1.0
            init_lambda = varargin{i+1};
        case 'initmaxcg'
            init_maxcg = varargin{i+1};
            % The maximum number of CG interations grows and shrinks.  This 'initmaxcg' is the initial number of maximum CG iterations on the first
            % pass.  Not sure why this is here, really.  See 'lowestmaxcg'.
            assert ( false, 'No longer implemented');
        case 'highestmaxcg'
            % The maximum number of CG iterations that the optimizer will let the CG iterations grow to.  There tends to be a lot of grinding if this
            % is too high, but I've noticed that if it's not high enough, then maybe you won't find the very best optimization values.  Your call.
            % Default = 300
            highest_maxcg = varargin{i+1};
        case 'lowestmaxcg'
            % The maximum number of CG interations grows and shrinks.  This 'initmaxcg' is the initial number of maximum CG iterations, which is also
            % used to reset after an HF iteration fails to find a better objective function value.  Default = 20
            lowest_maxcg = varargin{i+1};
        case 'cgtol'
            % This is some kind of relative tolerance used as a CG iteration stopping condition.  See Martens Deep Learning paper.  Default = 1e-29
            cgtol = varargin{i+1};
        case 'cgbtobjfun'  
            % A string of 'train' or 'test', which determines which dataset to evaluate the CG backtracking on.  You want 'train'.  Default = 'train'
            cgbt_objfun = varargin{i+1};
        case 'optevalfun'
            % This is important for certain types of optimizations that requires additional information.  The interface is 
            %
            % all_optional_args = optional_eval_fun(net, simparams, funs, v_inputtrain_T, v_inputtest_t, m_targettrain_T, m_targettest_t, all_simdata);    
            %
            % And these 'all_optional_args' are passed around to the evaluation functions.  Its for is 
            % all_optional_args{TvV_T}(trial_id) is a user defined structure, where TvV_T = 1 or 2, for either training or test, respectively, and 
            % trial_id is the gives the trial.  Thus there is a cell array for training vs. validation, and inside is a structure array.            
            %
            % An example of a reasonble eval fun parameter to supply is a random noise seed for each trial. 
            %
            % The difference between all_simdata and all_optional_args is that simdata is controlled by the functions that create the inputs and
            % targets, and also controlled by the forward pass function.  The all_optional_args is called once per iteration and set by a user defined
            % function.                                    
            optional_eval_fun = varargin{i+1};
        case 'optplotfun'
            % This is the plot function supplied by the user that the optimizer calls at the end of each HF iteration.  It's interface is given by 
            %
            % plot_stats = optional_plot_fun(net, simparams, funs, cg_found_better_solution, f2, random_trial_idxs, forward_pass_T, forward_pass_s, ...           
            %                                v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
            %                                m_targettrain_T, m_targettrain_s, m_targettest_t, all_optional_args, all_simdata, all_plot_stats);
            %
            % plot_stats is some structure that gets tacked onto the stats struct that the optimizer saves.  
            optional_plot_fun = varargin{i+1};
        case 'rollinginitdecay'
            % This is a parameter used in Martens Deep Learning paper.   I haven't really messed with it, and things are always working.  Default =
            % 0.95;
            last_w_decay_factor = varargin{i+1};
        case 'dolinesearch'
            % Do you want to do a simple line search when the CG back tracking finishes?  If so, the answer is yes.  Basically, sometimes it helps a
            % little but it's not really necessary.
            do_line_search = varargin{i+1};           
        case 'userdefinedlearningrate'
            % Don't mess with this.  It's off by default.
            assert ( false, 'Stopped');
            do_user_defined_learning_rate = true;            
            user_defined_learning_rate = varargin{i+1};
        case 'userdefinedlearningratedecay'
            % Don't mess with this.  It's off by default.
            assert ( false, 'Stopped');
            user_defined_learning_rate_decay = varargin{i+1};            
        case 'simdata'
            % This is another hook for users to add data to make the optimizer more flexible.  As you read above, the order is to generate the inputs,
            % and then there were some extra steps before the targets were defined.  This simdata is intended to bridge that gap.  If you need to
            % store stuff in simdata to correctly implement targets, then this is the place to do it.  Simdata is also update by the lower level
            % scripts that call the forward pass, so this is how one might change a target depending on whether or not some condition during the
            % forward pass was met.
            % all_simdata has the same form above as all_optional_args... it's a cell array of length 2, 1 = training, 2 = validation.  Inside each
            % cell is a structure array, one element for each trial.
            % 
            % The difference between all_simdata and all_optional_args is that simdata is controlled by the functions that create the inputs and
            % targets, and also controlled by the forward pass function.  The all_optional_args is called once per iteration and set by a user defined
            % function.            
            all_simdata = varargin{i+1};
        case 'S'
            % The size of the minibatch used during the CG iterations.  Default is T (number of trials) / 5.0
            Suser = varargin{i+1};
        case 'Sfrac'
            Sfrac_user = varargin{i+1};
        case 'saveevery'
            % The optimizer will save the full state of the optimization every 'save_every' iterations.  It's highly recommended that you save every
            % now and again because optimization of deep architectures can take awhile, and things may break.  Default = 10
            save_every = varargin{i+1};
        case 'savepath'
            % This is the absolute path to save the snapshots of the optimization.  Default = './'
            save_path = varargin{i+1};
        case 'filenamepart'
            % This is a string that will help differentiate between the many optimizations that you will inevitably create.
            filename_part = varargin{i+1};
        case 'samplestyle'
            % 'random_rows' vs. 'random_blocks' See discussion above.  Default = 'random_rows'
            sample_style = varargin{i+1};
        case 'dosavethetas'  % Be careful with size!
            % You don't really want this.  It saves the parameters at each iteration.  Keep in mind that the parameters are already snapshotted to a
            % file every 'saveevery' HF iterations.  This is just piling on, in case you want to study something special.
            do_save_thetas = varargin{i+1};            
        case {'paramstosave', 'simparams'}
            % Simparams are user defined parameters.  Unlike the optional_val_fun, which supplies the all_optional_args cell array, this is simply a
            % structure of parameters that can be used by the subroutines of the optimizter, e.g. rnn_hf_allfun2.m
            simparams = varargin{i+1};   % Nothing done with these for training, simply saved with the network and stats structure at every save.
        case 'trainfilename'
            % This is useful, it will slurp up a copy of the file supplied by this user parameter and save it as part of the stats structure that is 
            % saved.  If that file is function that called this optimizer, then you can save all the parameters and the exact training file that 
            % executed the optimiztion.  My experience is that this is often necessary to reconstruct exactly what I did.
            train_file_name = varargin{i+1};
        case 'doparallelnetwork'
            % Should the forward pass be evaluated in parallel or serially?  Default = false (serially)
            do_parallel_network = varargin{i+1};
        case 'doparallelobjfun'
            % Should the objective function be evaluated in parallel or serially? Default = false (serially)  (may be antiquated)
            do_parallel_objfun = varargin{i+1};
        case 'doparallelgradient'
            % Should the gradient be evaluated in parallel or serially?  Default = false (serially)
            do_parallel_gradient = varargin{i+1};
        case 'doparallelcgafun'
            % Should the CG iterations be evaluated in parallel or serially?  Default = false (serially)
            do_parallel_cg_afun = varargin{i+1};
        case 'doparallelprecon'
            % Should the preconditioner be evaluated in parallel or serially? (RNN doesn't use one). Default = false (serially)
            do_parallel_cg_afun = varargin{i+1};
        case 'dowrapperscgafun'
            % Don't use this.
            assert ( false, 'Stopped');
            do_wrappers_cg_afun = varargin{i+1};
        case 'dowrappersnetwork'
            % Don't use this.
            assert ( false, 'Stopped');
            do_wrappers_network = varargin{i+1};
        case 'dowrappersobjfun'
            % Don't use this.
            assert (false, 'Stopped');
            do_wrappers_objfun = varargin{i+1};
        case 'dowrappersgradient'
            % Don't use this.
            assert ( false, 'Stopped');
            do_wrappers_gradient = varargin{i+1};
        case 'dowrappersprecon'
            % Don't use this.
            assert ( false, 'Stopped');
            do_wrappers_cg_afun = varargin{i+1};            
        otherwise
            assert ( false, [' Variable argument ' varargin{i} ' not recognized.']);
    end
end

assert (do_line_search == false || do_user_defined_learning_rate == false, 'Can''t have both line search and learning rate');
assert ( rho_drop_thresh > rho_boost_thresh, 'Stopped.');
assert ( rho_boost_val > rho_drop_val, 'Stopped');


if ~isempty(train_file_name)
    [status, train_file_contents] = unix(['cat ' train_file_name]);
    assert(status == 0, ['Error loading training file contents: ' train_file_name '.']);
end

input_type = 'numeric';
if  isnumeric( v_inputtrain_T_start )
    input_type = 'numeric';
    using_cells = 0;
    
    v_inputtrain_T = v_inputtrain_T_start;
    m_targettrain_T = m_targettrain_T_start;
    v_inputtest_t = v_inputtest_t_start;
    m_targettest_t = m_targettest_t_start;
    if size(v_inputtest_t,2) == 0
        do_validation = false;
    else
        do_validation = true;
    end
elseif iscell( v_inputtrain_T_start )
    input_type = 'cell';
    using_cells = 1;
    
    v_inputtrain_T = v_inputtrain_T_start;
    m_targettrain_T = m_targettrain_T_start;
    v_inputtest_t = v_inputtest_t_start;
    m_targettest_t = m_targettest_t_start;
    if size(v_inputtest_t,2) == 0
        do_validation = false;
    else
        do_validation = true;
    end       
elseif isa(v_inputtrain_T_start, 'function_handle') 
    % start with a function handle that returns a cell of trials, cuz
    % that's what I need first. -DCS:2011/10/03
    input_type = 'function';
    using_cells = 1;
    
    train_fun = v_inputtrain_T_start;
    test_fun = v_inputtest_t_start;
    do_inputs = true;
    do_targets = false;
    [v_inputtrain_T, ~, ~, net, all_simdata] = train_fun(net, {}, {}, simparams, all_simdata, do_inputs, do_targets);
    do_inputs = false;
    do_targets = true;
    [~, m_targettrain_T, ~, net, all_simdata] = train_fun(net, v_inputtrain_T, {}, simparams, all_simdata, do_inputs, do_targets);
    if ~isempty(test_fun)
        do_validation = true;
        do_inputs = true;
        do_targets = false;
        [v_inputtest_t, ~, ~, net, all_simdata] = test_fun(net, {}, {}, simparams, all_simdata, do_inputs, do_targets);
        do_inputs = false;
        do_targets = true;
        [~, m_targettest_t, ~, net, all_simdata] = test_fun(net, v_inputtest_t, {}, simparams, all_simdata, do_inputs, do_targets);
    else
        do_validation = false;
        v_inputtest_t = [];
        m_targettest_t = [];
    end
end

if ~isempty(v_inputtrain_T);
    [~,T] = size(v_inputtrain_T);
elseif ~isempty(m_targettrain_T)
    [~,T] = size(m_targettrain_T);
end
assert ( T > 0, 'stopped');
if isempty(v_inputtrain_T)
    v_inputtrain_T = cell(1,T);
end
if isempty(m_targettrain_T)
    m_targettrain_T = cell(1,T);
end

if do_validation
    [~,t] = size(v_inputtest_t);
else
    t = 0;
end

assert (isnan(Suser) || isnan(Sfrac_user), 'Can only use one.');
assert (~isnan(Suser) || ~isnan(Sfrac_user), 'Must only use one.');
if isnan(Suser)
    %user_specified_S = false;
    do_user_specified_S_frac = true;
    S_frac = Sfrac_user;
end
if isnan(Sfrac_user)
    %user_specified_S = true;
    do_user_specified_S_frac = false;
    S_frac = NaN;
end
if do_user_specified_S_frac
    S = ceil(T*S_frac);
else
    S = Suser;
end

if isempty(all_simdata)
    for i = 1:T
        simdata_T(i).id = i;
    end
    all_simdata{1} = simdata_T;
    
    if t > 0 
        for i = 1:t
            simdata_t(i).id = i;
        end
        all_simdata{2} = simdata_t;
    else
        all_simdata{2} = struct;
    end
end

lambda = init_lambda; % note if init_lambda is 0, then lambda will always stay zero because all modifications are multiplications

assert ( T > 0, 'There are zero dimensions in the data.');
%assert ( t > 0, 'Should be able to handle this case down the road.' );
if ( t > 0 )
    do_validation = true;
else
    do_validation = false;
end
assert ( S <= T, 'The number of samples should be less than or equal to the number of data points or trials.');
assert ( S > 0, 'Minibatch size needs to actually pick samples.');
% This function will still be application specific, but getting closer to an abstract function.

if isempty(filename_part)
    disp('Not saving intermediate HF solutions at each iteration because a filename part was not given.');
else
    disp(['Saving intermediate HF solutions at each iteration in directory ' save_path ' with filename part ' filename_part '.']);
end


if do_parallel_network
    disp('Evaluating the network function in parallel');
    if ( do_wrappers_network )
        disp('... and using wrappers.');
    end
else
    disp('Evaluating the network function serially.');
end
if do_parallel_objfun
    disp('Evaluating the objective function in parallel');
    if do_wrappers_objfun
        disp('... and using wrappers.');
    end
else
    disp('Evaluating the objective function serially.');
end
if do_parallel_gradient
    disp('Evaluating the gradient function in parallel');
    if do_wrappers_gradient
        disp('... and using wrappers.');
    end
else
    disp('Evaluating the gradient function serially.');
end
if do_parallel_cg_afun
    disp('Evaluating the CG Afun function in parallel');
    if do_wrappers_cg_afun
        disp('... and using wrappers.');
    end
else
    disp('Evaluating the CG Afun function serially.');
end



switch lower(cgbt_objfun)
    case 'train'
        do_eval_cg_train = 1;
    case 'test'
        do_eval_cg_train = 0;
    otherwise
        assert (false, ['Don''t recognize ' cgbt_objfun]);
end

if ( do_plot )
    f1 = figure;
    if do_plot_all_objectives
        f2 = figure;
    end
    f3 = figure;
end

switch lower(net_type)
    case 'dn' % Deep Network - an arbitrarily long feed-forward neural network.
        eval_network_dn2 = create_eval_network_dn2(weight_cost);
        eval_objfun_dn2 = create_eval_objfun_dn2(weight_cost);
        eval_objfun_with_network_dn2 = create_eval_objfun_with_network_dn2(weight_cost);
        eval_gradient_dn2 = create_eval_gradient_dn2(weight_cost);
        eval_cg_afun_dn2 = create_eval_cg_afun_dn2(weight_cost);
        eval_preconditioner_dn2 = create_eval_preconditioner_dn2(weight_cost);
        eval_gradient_with_network_dn2 = create_eval_gradient_with_network_dn2(weight_cost);
        
        
        eval_network = create_eval_network2(eval_network_dn2, weight_cost);
        eval_objfun = create_eval_objfun2(eval_objfun_dn2, weight_cost);
        eval_objfun_with_network = create_eval_objfun_with_network2(eval_objfun_with_network_dn2, weight_cost);
        eval_gradient = create_eval_gradient2(eval_gradient_dn2, weight_cost);
        eval_cg_afun = create_eval_cg_afun2(eval_cg_afun_dn2, weight_cost);
        eval_preconditioner = create_eval_preconditioner2(eval_preconditioner_dn2, weight_cost);
        eval_gradient_with_network = create_eval_gradient_with_network2(eval_gradient_with_network_dn2, weight_cost);

    case 'test2' % like shock-g said, do whatcha like
        eval_network_test2 = create_eval_network_test2(weight_cost);
        eval_objfun_test2 = create_eval_objfun_test2(weight_cost);
        eval_gradient_test2 = create_eval_gradient_test2(weight_cost);
        eval_cg_afun_test2 = create_eval_cg_afun_test2(weight_cost);
        % New
        %eval_objfun_and_network_test2 = create_eval_objfun_and_network_test2(weight_cost);
        eval_gradient_with_network_test2 = create_eval_gradient_with_network_test2(weight_cost);
        eval_objfun_with_network_test2 = create_eval_objfun_with_network_test2(weight_cost);
        
        eval_network = create_eval_network2(eval_network_test2, weight_cost);
        eval_objfun = create_eval_objfun2(eval_objfun_test2, weight_cost);
        eval_gradient = create_eval_gradient2(eval_gradient_test2, weight_cost);
        eval_cg_afun = create_eval_cg_afun2(eval_cg_afun_test2, weight_cost);
        eval_preconditioner = [];
        % New
        %eval_objfun_and_network = create_eval_objfun_and_network2(eval_objfun_and_network_test2, weight_cost);
        eval_objfun_with_network = create_eval_objfun_with_network2(eval_objfun_with_network_test2, weight_cost);
        eval_gradient_with_network = create_eval_gradient_with_network2(eval_gradient_with_network_test2, weight_cost);

    case 'rnn-trials' % Recurrent neural network with trial structure, uses the parallel toolbox.
        % These anonymous functions take a snapshot of the workspace at the
        % time they were created. If there is data that is a large matrix, then the anonymous function stores that
        % large matrix inside it. See message 14 in this thread:
        % http://www.mathworks.com/matlabcentral/newsreader/view_thread/235926
        eval_network_rnn2 = create_eval_network_rnn2(weight_cost);
        eval_objfun_rnn2 = create_eval_objfun_rnn2(weight_cost);
        eval_gradient_rnn2 = create_eval_gradient_rnn2(weight_cost);
        eval_cg_afun_rnn2 = create_eval_cg_afun_rnn2(weight_cost);
        eval_preconditioner_rnn2 = [];         %#ok<NASGU>
        % New
        %eval_objfun_and_network_rnn2 = create_eval_objfun_and_network_rnn2(weight_cost);
        eval_objfun_with_network_rnn2 = create_eval_objfun_with_network_rnn2(weight_cost);
        eval_gradient_with_network_rnn2 = create_eval_gradient_with_network_rnn2(weight_cost);
        
        eval_network = create_eval_network2(eval_network_rnn2, weight_cost);
        eval_objfun = create_eval_objfun2(eval_objfun_rnn2, weight_cost);
        eval_gradient = create_eval_gradient2(eval_gradient_rnn2, weight_cost);
        eval_cg_afun = create_eval_cg_afun2(eval_cg_afun_rnn2, weight_cost);
        eval_preconditioner = [];
        % New
        %eval_objfun_and_network = create_eval_objfun_and_network2(eval_objfun_and_network_rnn2, weight_cost);
        eval_objfun_with_network = create_eval_objfun_with_network2(eval_objfun_with_network_rnn2, weight_cost);
        eval_gradient_with_network = create_eval_gradient_with_network2(eval_gradient_with_network_rnn2, weight_cost);
        
        
    case 'mrnn-trials' % Multiplicative recurrent neural network with trial structure, uses the parallel toolbox.
        % These anonymous functions take a snapshot of the workspace at the
        % time they were created. If there is data that is a large matrix, then the anonymous function stores that
        % large matrix inside it. See message 14 in this thread:
        % http://www.mathworks.com/matlabcentral/newsreader/view_thread/235926
        eval_network_mrnn2 = create_eval_network_mrnn2(weight_cost);
        eval_objfun_mrnn2 = create_eval_objfun_mrnn2(weight_cost);
        eval_gradient_mrnn2 = create_eval_gradient_mrnn2(weight_cost);
        eval_cg_afun_mrnn2 = create_eval_cg_afun_mrnn2(weight_cost);
        eval_preconditioner_mrnn2 = [];         %#ok<NASGU>
        % New        
        eval_objfun_with_network_mrnn2 = create_eval_objfun_with_network_mrnn2(weight_cost);
        eval_gradient_with_network_mrnn2 = create_eval_gradient_with_network_mrnn2(weight_cost);
        
        eval_network = create_eval_network2(eval_network_mrnn2, weight_cost);
        eval_objfun = create_eval_objfun2(eval_objfun_mrnn2, weight_cost);
        eval_gradient = create_eval_gradient2(eval_gradient_mrnn2, weight_cost);
        eval_cg_afun = create_eval_cg_afun2(eval_cg_afun_mrnn2, weight_cost);
        eval_preconditioner = [];
        % New
        %eval_objfun_and_network = create_eval_objfun_and_network2(eval_objfun_and_network_rnn2, weight_cost);
        eval_objfun_with_network = create_eval_objfun_with_network2(eval_objfun_with_network_mrnn2, weight_cost);
        eval_gradient_with_network = create_eval_gradient_with_network2(eval_gradient_with_network_mrnn2, weight_cost);                                     
        
    otherwise
        disp(['Unknown network type ' net_type '.']);
        assert ( false, 'Case not implemented yet.');
end


funs.evalNetwork = eval_network;
funs.evalObjfun = eval_objfun;
funs.evalGradient = eval_gradient;
funs.evalCGAFun = eval_cg_afun;
funs.evalPreconditioner = eval_preconditioner;
funs.evalOptEvalFun = optional_eval_fun;

if ( do_eval_cg_train )
    disp('Using training data for CG backtracking.');
else
    disp('Using test data for CG backtracking.');
end

if ~using_cells
    disp(['Using a minibatch size of ' num2str(S) ' out of ' num2str(T) ' total samples.']);
else
    disp(['Using a minibatch size of ' num2str(S) ' out of ' num2str(T) ' total trials.']);
end

piter = 0;
go = 1;				% Allow go number of increases in the error count, will show overfitting, but
% we'll be sure about the bottom.

init_maxcg = lowest_maxcg;
maxcg = init_maxcg;
ncgiters_constant_decreasing = 0;  % There's only so much bad CG a man can take.
disp(['Init max CG iterations: ' num2str(maxcg) ]);


all_thetas = []; 
if do_save_thetas  % Be careful here!
    all_thetas = zeros(length(net.theta), max_hf_iters);
end

pn_cgstart = zeros(size(net.theta));
%theta_last = net.theta;

hf_iter = 0;
total_hf_consecutive_suck_count = 0;
total_hf_suck_count = 0;

% new_data_objfun = Inf;
% objfun = Inf;
objfun_last = Inf;
%objfun_last_allobjs = Inf;
% objfun_train = Inf;
objfun_test = Inf;
objfun_test_allobjs = Inf;
% objfun_cg = Inf;
% objfun_cg_train = Inf;
objfun_cg_test = Inf;
objfun_cg_test_allobjs = Inf;
% objfun_cg_forward = Inf;
% objfun_cg_forward_other = Inf;
objfun_ls_train_min_allobjs = Inf; %#ok<NASGU>
objfun_train_allobjs = Inf;

grad_norm = Inf;
rho = NaN;

% Stuff for stats
all_lambdas = [];
all_grad_norms = [];
all_rhos = [];
all_objfun_trains = [];
all_objfun_trains_allobjs = [];
all_objfun_tests = [];
all_objfun_tests_allobjs = [];
all_hf_iter_times = [];
all_cg_iters_taken = [];
all_cg_iters_computed = [];
all_cg_found_better_solutions = [];
all_plot_stats = [];

do_recompute_gradient = true;
do_resample_data = true;
do_recompute_rho = true; %#ok<NASGU>
%cg_found_better_solution = false;

hf_iter_time = 0.0; %#ok<NASGU>
total_time = 0.0;
stop_string = '';

TvV_T = 1; % Training vs. validation (training)
TvV_V = 2; % Training vs. validation (validation)

exit_flag = NaN;

% Have to make sure that the we first evaluate the network to make sure we don't go backwards on the first trial.

all_train_trial_idxs = 1:T;
all_test_trial_idxs = 1:t;
if ( ~isempty(optional_eval_fun) )
    if ( ~using_cells )
        all_optional_args = optional_eval_fun(net, simparams, funs, v_inputtrain_T, v_inputtest_t, m_targettrain_T, m_targettest_t, all_simdata);
    else
        all_optional_args = optional_eval_fun(net, simparams, funs, v_inputtrain_T, v_inputtest_t, m_targettrain_T, m_targettest_t, all_simdata);
        
    end
end

package = eval_objfun(net, v_inputtrain_T, m_targettrain_T, TvV_T, all_train_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
objfun_train = package{1};
if isnan(objfun_train)
    objfun_train = Inf;
end
all_simdata = package{end};
if ( do_validation > 0 )
    package = eval_objfun(net, v_inputtest_t, m_targettest_t, TvV_V, all_test_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
    objfun_test = package{1};
    all_simdata = package{end};
end
if ( do_eval_cg_train )
    objfun = objfun_train;
else
    assert ( do_validation, 'Can''t use test set for CG evaluation if there isn''t one!');
    objfun = objfun_test;
end
objfun_constant_decreasing = realmax;  % Catchs a case where lambda isn't small enough.

if display_level > 0
    if ( do_validation )
        disp(['Initial Objective function (train): ' num2str(objfun_train), ', Objective function (test): ', num2str(objfun_test)]);
    else
        disp(['Initial Objective function (train): ' num2str(objfun_train) '.']);
    end
end

while go
    if do_time_hf_iters
        tic;
    end
    
    hf_iter = hf_iter + 1;
    
    %%% Hack    
    if ~isinf(nhfiters_to_reset_matlabpool) && mod(hf_iter, nhfiters_to_reset_matlabpool) == 1
        warning('hacking the matlabpool memory leak by reseting');
        matlabpool('close');
        matlabpool('local');
    end
    
    if ( hf_iter > max_hf_iters )
        hf_iter = hf_iter - 1;  % For saving at the end, outside of the loop.
        stop_string = ['Stopping because total number of HF iterations is greater than ' num2str(max_hf_iters) '.'];
        exit_flag = 0; %#ok<NASGU>
        break;
    end
    
    if display_level > 0 
        disp(['HF iteration: ' num2str(hf_iter) '.']);
    end
    simparams.HFIter = hf_iter;  %%% XXX HACK, because I'm randomly adding this to simparams!  DCS 11/11/2012
    
    % Get the samples for the Hessian, this matlab notation should handle both matrices and cells.
    if ( do_resample_data )
        
        switch input_type
            case 'function'
                do_inputs = true;
                do_targets = false;
                % Note changing net is about passing back extra data, NOT changine the parameters.
                [v_inputtrain_T, ~, did_change_train_data, net, all_simdata] = train_fun(net, v_inputtrain_T, m_targettrain_T, simparams, all_simdata, do_inputs, do_targets); % can keep some old data.
                [~,T] = size(v_inputtrain_T);  
                if do_user_specified_S_frac
                    S = ceil(T * S_frac);
                    assert ( S > 0, 'The CG minibatch size must be greater than 0' );
                end
                m_targettrain_T  = cell(1,T);  % Really about deleting old stuff and keeping the sizes the same.
                if ( do_validation )
                    [v_inputtest_t, ~, did_change_test_data, net, all_simdata] = test_fun(net, v_inputtest_t, m_targettest_t, simparams, all_simdata, do_inputs, do_targets); %  can keep some old data.
                    [~,t] = size(v_inputtest_t);
                    m_targettest_t = cell(1, t); % Really about deleting old stuff and keeping the sizes the same.
                end
            case {'numeric', 'cell'}
                1;  % fall through, no problem.
            otherwise
                assert ( false, 'NIY');
        end
        
        
        % An optional function computed before each HF iteration to give the
        % network implementation a chance to compute some EXTRA STUFF to PASS
        % to the EVALUATION functions.  For example, computing condition
        % dependent initial conditions.
        if ~isempty(optional_eval_fun)
            all_optional_args = optional_eval_fun(net, simparams, funs, v_inputtrain_T, v_inputtest_t, m_targettrain_T, m_targettest_t, all_simdata);
        end
        
        
        % Now let's get onto the business of training.  We first evaluate the forward pass to be used for the gradient and the 2nd approximation.
        % Crucially, there is one and only one forward pass.  When we start evalutating CG backtracking to get the best update, those forward passes are
        % revaluated because the network has changed.  But up until that point, everybody is seeing the same single forward pass instantiation.
        package = eval_network(net, v_inputtrain_T, m_targettrain_T, TvV_T, all_train_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
        forward_pass_T = package{1};
        all_simdata = package{end};
        clear package;
        
        % Now compute the targets, simdata will be updated from the forward run of the network just above.
        switch input_type
            case 'function'
                do_inputs = false;
                do_targets = true;
                % Note changing the net structure is about passing back extra data, NOT changing the parameters.  Actually, just don't do it.
                [~, m_targettrain_T, did_change_train_data, net, all_simdata] = train_fun(net, v_inputtrain_T, m_targettrain_T, simparams, all_simdata, do_inputs, do_targets); % can keep some old data.
                if ( do_validation )
                    [~, m_targettest_t, did_change_test_data, net, all_simdata] = test_fun(net, v_inputtest_t, m_targettest_t, simparams, all_simdata, do_inputs, do_targets); %  can keep some old data.
                    %[~,t] = size(m_targettest_t);
                end
            case {'numeric', 'cell'}
                1;  % fall through, no problem.
            otherwise
                assert ( false, 'NIY');
        end
        
        
        % Can't evaluate the objective function until the targets are evaluated.  Can't evaluate the targets until the
        % forward pass is evaluated.  Can't evaluate the forward pass until the inputs are evaluated.
        package = eval_objfun_with_network(net, v_inputtrain_T, m_targettrain_T, forward_pass_T, TvV_T, all_train_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
        new_data_objfun = package{1};
        if isnan(new_data_objfun)
            new_data_objfun = Inf;
        end
        all_simdata = package{end};
        if ( do_validation )
            package = eval_objfun(net, v_inputtest_t, m_targettest_t, TvV_V, all_test_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
            new_data_objfun_test = package{1};
            all_simdata = package{end};
        end
        
        if display_level > 0 
            if do_validation && did_change_train_data && did_change_test_data
                disp(['New Dataset Objfun (train): ' num2str(new_data_objfun) ', Objfun (test): ' num2str(new_data_objfun_test) '.']);
            elseif did_change_train_data
                disp(['New Dataset Objfun (train): ' num2str(new_data_objfun) '.']);
            end
        end
        
        % Do CG.
        % Set up the gradient, denoted b, note it's the full gradient based on the entire training set.
        % I think the fact that the gradient is evaluated on the entire dataset allows the optimization to add,
        % consecutively, and without an annealing schedule, the CG output as total solution.  This is why in the fake-RLS
        % implementations, we had to be careful about what we added up, because the gradient was evaluated only on the
        % cases of the moment.  This is a very important feature of the optimization.
        
        % Eval the network once for the CG iterates.  (note this loop was in the "while isempty(all_pn)", which couldn't have been correct... I think.
        % Get the subsample for the CG evaluation.
        %forward_pass_s = eval_network(net, v_inputtrain_s, m_targettrain_s, train_vs_valid__train, random_trial_idxs, all_optional_args, 'doparallel', do_parallel_network, 'dowrappers', do_wrappers_network);
        
        switch sample_style
            case 'random_rows'
                if ( S < T )
                    rp = randperm(T);			% random data
                    random_trial_idxs = rp(1:S);			% S random peices of data
                else
                    random_trial_idxs = all_train_trial_idxs;
                end
                v_inputtrain_s = v_inputtrain_T(:,random_trial_idxs);
                if ~isempty(m_targettrain_T)
                    m_targettrain_s = m_targettrain_T(:,random_trial_idxs);
                else
                    m_targettrain_s = [];
                end

            case 'random_blocks'
                if ( S < T )
                    start_idx = randi(T-S);
                    stop_idx = start_idx + S - 1;
                    random_trial_idxs = start_idx:stop_idx;
                else
                    random_trial_idxs = all_train_trial_idxs;
                end
                v_inputtrain_s = v_inputtrain_T(:,random_trial_idxs);
                if ~isempty(m_targettrain_T)
                    m_targettrain_s = m_targettrain_T(:,random_trial_idxs);
                else
                    m_targettrain_s = [];
                end
                                
            otherwise
                assert ( false, ['Sample style: ' sample_style ' not supported.']);
        end
    end
    
    forward_pass_s = forward_pass_T(random_trial_idxs);  % Now the full pass is guarenteed to be there.
    
    if ( ~isempty(eval_preconditioner) )
        package = eval_preconditioner(net, v_inputtrain_s, m_targettrain_s, lambda, TvV_T, random_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_precon, 'dowrappers', do_wrappers_precon);
        precon = package{1};
        all_simdata = package{end};
    else
        precon = [];
    end
    
    if ( do_recompute_gradient )
        if do_grad_on_full_data
            package = eval_gradient_with_network(net, v_inputtrain_T, m_targettrain_T, forward_pass_T, TvV_T, all_train_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_gradient, 'dowrappers', do_wrappers_gradient);
            grad = package{1};
            grad_norm = norm(grad);
            all_simdata = package{end};
            clear package;
        else
            assert (false, 'no longer supported.');
        end
        %b = -grad; % negative gradient
    end
    
    
    
    all_pn = {};
    cg_gamma_idxs = [];
    % this just protects against accidental negative curvature (ie. -1e10 or something)
    nlambda_increases_for_negative_curvature = 0;
    while isempty(all_pn) && nlambda_increases_for_negative_curvature < max_lambda_increases_for_negative_curvature
        
        [all_pn, cg_gamma_idxs, all_cg_phis, pAp, all_simdata] = ...
            conjgrad_2(@(dw, all_simdata_p) eval_cg_afun(net, v_inputtrain_s, m_targettrain_s, dw, lambda, forward_pass_s, TvV_T, random_trial_idxs, all_optional_args, all_simdata_p, 'doparallel', do_parallel_cg_afun, 'dowrappers', do_wrappers_cg_afun), ...
            -grad, pn_cgstart, maxcg, lowest_maxcg, precon, all_simdata, 'epsilon', cgtol, 'displaylevel', display_level);
        
        if ( pAp < smallest_pAp )
            smallest_pAp = pAp;
        end
        
%         if ( pAp <= 0.0 )		% protection against tiny negative curvature.
%             
%             nlambda_increases_for_negative_curvature = nlambda_increases_for_negative_curvature + 1;
%             
%             if ( lambda > realmin )	% really 0.0, so increasing lambda won't help.
%                 lambda = rho_boost_val * lambda;
%             end
%             do_resample_data = 0;	%#ok<NASGU> % if lambda is very small, this may be the only way to get out of
%             % trouble. but not used! -DCS:2011/10/07
%             all_pn = {};
%             disp(['Found non-positive curvature, increasing lambda to ' num2str(lambda) '.']);
%         end
    end
    % Always check for negative curvature, we try to increase lambda, to forgive slightly buggy implementations, but
    % sometimes that just doesn't fly. The other stop conditions are below, but this is a show stopper so check
    % immediately.
    if ( pAp < 0.0 )
        stop_string = 'Stopping because negative curvature was found.';
        exit_flag = -14;  %#ok<NASGU> % no fminunc equivalent.
        go = 0; %#ok<NASGU>
        break;
    end
    if ( pAp == 0.0 )
        stop_string = 'Stopping because pAp is identically zero.  I think this is a success condition.';
        exit_flag = 11;  %#ok<NASGU> % no fminunc equivalent.
        go = 0; %#ok<NASGU>
        break;
    end
    
    
    % CG Backtracking
    % Now do some CG backtracking according to our error function and not what the min residual is.
    pn = all_pn{end};
    cgtheta = net.theta + pn; 	% net holds best solution from last HF iteration.
    cgnet = net;
    cgnet.theta = cgtheta;
    
    package = eval_objfun(cgnet, v_inputtrain_T, m_targettrain_T, TvV_T, all_train_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
    objfun_cg_train = package{1};
    objfun_cg_train_allobjs = package{2};
    all_simdata = package{end};
    if ( do_validation )
        package = eval_objfun(cgnet, v_inputtest_t, m_targettest_t, TvV_V, all_test_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
        objfun_cg_test = package{1};
        objfun_cg_test_allobjs = package{2};
        all_simdata = package{end};
    end
    if ( do_eval_cg_train )
        objfun_cg = objfun_cg_train;
        objfun_cg_allobjs = objfun_cg_train_allobjs;
        objfun_cg_forward = objfun_cg_train;
        objfun_cg_forward_allobjs = objfun_cg_train_allobjs;
        objfun_cg_forward_other = objfun_cg_test;
        objfun_cg_forward_other_allobjs = objfun_cg_test_allobjs;
    else
        objfun_cg = objfun_cg_test;
        objfun_cg_allobjs = objfun_cg_test_allobjs;
        objfun_cg_forward = objfun_cg_test;
        objfun_cg_forward_allobjs = objfun_cg_test_allobjs;
        objfun_cg_forward_other = objfun_cg_train;
        objfun_cg_forward_other_allobjs = objfun_cg_train_allobjs;
    end
    
    if display_level > 1
        if ( do_validation )
            disp(['CG Objfun (train): ' num2str(objfun_cg_train) ', CG Objfun (test): ' num2str(objfun_cg_test) ' at iter: ' num2str(cg_gamma_idxs(length(all_pn)))]);
        else
            disp(['CG Objfun (train): ' num2str(objfun_cg_train) ' at iter: ' num2str(cg_gamma_idxs(length(all_pn)))]);
        end
    end
    cgbt_min_idx = length(all_pn);
    last_cgbt_eval_iter = length(all_pn);
    cg_found_better_solution = 0;
    cg_bt_did_break = 0;
    cg_did_increase = 0; % did the objective function increase as CG went forward in time? I.e. if not, then we may
    % need to continue going forward.
    for i = length(all_pn)-1:-1:1
        
        cgnet = net;
        pn = all_pn{i};
        cgtheta = net.theta + pn;	% net holds best solution from last HF iteration.
        
        cgnet.theta = cgtheta;
        
        package = eval_objfun(cgnet, v_inputtrain_T, m_targettrain_T, TvV_T, all_train_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
        objfun_cg_train = package{1};
        objfun_cg_train_allobjs = package{2};
        all_simdata = package{end};
        if ( do_validation )
            package = eval_objfun(cgnet, v_inputtest_t, m_targettest_t, TvV_V, all_test_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
            objfun_cg_test = package{1};
            objfun_cg_test_allobjs = package{2};
            all_simdata = package{end};
        end
        if ( do_eval_cg_train)
            objfun_cg = objfun_cg_train;
            objfun_cg_allobjs = objfun_cg_train_allobjs;
            objfun_cg_other = objfun_cg_test;
            objfun_cg_other_allobjs = objfun_cg_test_allobjs;
        else
            objfun_cg = objfun_cg_test;
            objfun_cg_allobjs = objfun_cg_test_allobjs;
            objfun_cg_other = objfun_cg_train;
            objfun_cg_other_allobjs = objfun_cg_train_allobjs;
        end
        
        if display_level > 1
            if ( do_validation )
                disp(['CG Objfun (train): ' num2str(objfun_cg_train) ', CG Objfun (test): ' num2str(objfun_cg_test) ' at iter: ' num2str(cg_gamma_idxs(i))]);
            else
                disp(['CG Objfun (train): ' num2str(objfun_cg_train) ' at iter: ' num2str(cg_gamma_idxs(i))]);
            end
        end
        if ( objfun_cg_forward < objfun_cg  && objfun_cg_forward < new_data_objfun)  %objfun ) % the forward one is the hopeful iter
            objfun_cg = objfun_cg_forward;
            objfun_cg_allobjs = objfun_cg_forward_allobjs;
            if ( do_eval_cg_train )
                objfun_cg_train = objfun_cg_forward;
                objfun_cg_train_allobjs = objfun_cg_forward_allobjs;
                objfun_cg_test = objfun_cg_forward_other;
                objfun_cg_test_allobjs = objfun_cg_forward_other_allobjs;
            else
                objfun_cg_test = objfun_cg_forward;
                objfun_cg_test_allobjs = objfun_cg_forward_allobjs;
                objfun_cg_train = objfun_cg_forward_other;
                objfun_cg_train_allobjs = objfun_cg_forward_other_allobjs;
            end
            cgbt_min_idx = last_cgbt_eval_iter;
            cg_bt_did_break = 1;
            cg_found_better_solution = 1;
            break;
        end
        if ( objfun_cg_forward > objfun_cg )
            cg_did_increase = 1;
        end
        cg_bt_did_break = 0;
        objfun_cg_forward = objfun_cg; % forward is the last evaluated objective function, but "forward" from CG perspective
        objfun_cg_forward_allobjs = objfun_cg_allobjs;
        objfun_cg_forward_other = objfun_cg_other;
        objfun_cg_forward_other_allobjs = objfun_cg_other_allobjs;
        last_cgbt_eval_iter = i;
    end
    if ( ~cg_bt_did_break  && objfun_cg < new_data_objfun ) % have to check case where CG iter 1 was still better than last HF iter.
        % was ~cg_bt_did_break & objfun_cg < objfun
        cg_found_better_solution = 1;
        cgbt_min_idx = 1;
    end
    
    % Now decide what to do based on the CG backtracking.
    if ( cg_found_better_solution )
        % Good case.  We found a CG iteration with lower objective function value
        cg_min_idx = cg_gamma_idxs(cgbt_min_idx);
        
        if display_level > 1
            disp(['Max CG iter: ' num2str(maxcg) ', CG Backtracking chose iter: ' num2str(cg_min_idx) '.']);
        end
        
        if display_level > 2
            if ( objfun_cg < new_data_objfun && objfun_cg > objfun )
                disp('Continuing because objfun was less than new dataset objective function');
                disp('   but greater than the old dataset objective function.');            
            end
        end
        
        total_hf_consecutive_suck_count = 0;
        
        %theta_last = net.theta;
        
        % Do a little line search.
        good_cg_incr = all_pn{cgbt_min_idx};        
        cgood = 1;
        %objfun_ls_init = objfun_cg;  % Value doesn't change.
        objfun_ls = objfun_cg;
        objfun_ls_allobjs = objfun_cg_allobjs; %#ok<NASGU>
        objfun_ls_min = objfun_cg;
        objfun_ls_min_allobjs = objfun_cg_allobjs;
        objfun_ls_train = objfun_cg_train; %#ok<NASGU>
        objfun_ls_train_allobjs = objfun_cg_train_allobjs; %#ok<NASGU>
        objfun_ls_test = objfun_cg_test;
        objfun_ls_test_allobjs = objfun_cg_test_allobjs;
        objfun_ls_train_min = objfun_cg_train;
        objfun_ls_train_min_allobjs = objfun_cg_train_allobjs;
        objfun_ls_test_min = objfun_cg_test;
        objfun_ls_test_min_allobjs = objfun_cg_test_allobjs;
        objfun_ls_last = realmax;
        if do_line_search
            lsnet = net;
            %nlinesearches = 16;
            %cs = linspace(0,1,nlinesearches+2);
            %cs = cs(2:end-1);
            %for i = nlinesearches:-1:1
            % c = cs(i);
            c = 0.98;
            i = 0;
            min_frac_ls_decrease = 0.01;
            c_decrease_val = 1.5;
            while objfun_ls < objfun_ls_last
                i = i+1;
                c = 1 - c_decrease_val*(1-c);
                if c < 0
                    break;
                end
                lsnet.theta = net.theta + c*good_cg_incr;
                objfun_ls_last = objfun_ls;
                package = eval_objfun(lsnet, v_inputtrain_T, m_targettrain_T, TvV_T, all_train_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
                objfun_ls_train = package{1};
                objfun_ls_train_allobjs = package{2};
                all_simdata = package{end};
                if ( do_validation )
                    package = eval_objfun(lsnet, v_inputtest_t, m_targettest_t, TvV_V, all_test_trial_idxs, all_optional_args, all_simdata, 'doparallel', do_parallel_objfun, 'dowrappers', do_wrappers_objfun);
                    objfun_ls_test = package{1};
                    objfun_ls_test_allobjs = package{2};
                    all_simdata = package{end};
                end
                if ( do_eval_cg_train)
                    objfun_ls = objfun_ls_train;
                    objfun_ls_allobjs = objfun_ls_train_allobjs;
                else
                    objfun_ls = objfun_ls_test;
                    objfun_ls_allobjs = objfun_ls_test_allobjs;
                end
                
                if objfun_ls < objfun_ls_min
                    objfun_ls_min = objfun_ls;
                    objfun_ls_min_allobjs = objfun_ls_allobjs;
                    objfun_ls_train_min = objfun_ls_train;
                    objfun_ls_train_min_allobjs = objfun_ls_train_allobjs;
                    objfun_ls_test_min = objfun_ls_test;
                    objfun_ls_test_min_allobjs = objfun_ls_test_allobjs;
                    cgood = c;
                end
                
                if display_level > 1 
                    if ( do_validation )
                        disp(['LS Objfun (train): ' num2str(objfun_ls_train) ', LS Objfun (test): ' num2str(objfun_ls_test) ' at iter: ' num2str(i)]);
                    else
                        disp(['LS Objfun (train): ' num2str(objfun_ls_train) ' at iter: ' num2str(i)]);
                    end
                end
                frac_ls_decrease = abs((objfun_ls_last - objfun_ls) / objfun_ls_last);
                if objfun_ls > objfun_ls_last  || frac_ls_decrease < min_frac_ls_decrease % Assume things decrease from cg solution to start.
                    break;
                end
            end
        elseif  do_user_defined_learning_rate
            % Completely crazy, but I guess it makes sense that sometimes a single HF iteration cannot sample the entire distribution            
            % So despite being a second order method, we can allow the user to have a user-defined learning rate.
            cgood = user_defined_learning_rate;
            if user_defined_learning_rate_decay < 1.0  % If > 1, will ignore, which is probably a good thing.
                user_defined_learning_rate = user_defined_learning_rate * user_defined_learning_rate_decay;
                if display_level > 1 
                    disp(['Learning rate: ' num2str(user_defined_learning_rate, 5)]);
                end
            end
        end
        net.theta = net.theta + cgood*good_cg_incr;
        
        objfun_last = objfun;
        if strcmp(input_type, 'function')
            objfun_last = new_data_objfun;
        end
        objfun = objfun_ls_min;
        objfun_allobjs = objfun_ls_min_allobjs;
        
        do_recompute_rho = true;
        do_recompute_gradient = true;
        do_resample_data = true;
        
        %%new_data_objfun = objfun; %#ok<NASGU>  %why would I do this renaming at end of iter?
        objfun_train = objfun_ls_train_min;	% get from best CG eval
        objfun_train_allobjs = objfun_ls_train_min_allobjs;
        objfun_test = objfun_ls_test_min;	% get from best CG eval
        objfun_test_allobjs = objfun_ls_test_min_allobjs;
        objfun_constant_decreasing = realmax;
        
        % lambda will be recompute using rho calculation
        if ~any(isnan(all_pn{end}))    % xxx small bug here.  Using the decay factor has nothing to do with isnan. DCS 5/21/2013
            pn_cgstart = all_pn{end};
        else
            pn_cgstart = last_w_decay_factor * all_pn{end};  % xx not modified by line search or anything
        end
        maxcg = ceil(cg_min_idx * cg_increase_factor);
        ncgiters_constant_decreasing = 0;
        
    elseif ( ~cg_did_increase && ~cg_bt_did_break && ~isinf(objfun_cg) && objfun_cg < objfun_constant_decreasing && ...
            ncgiters_constant_decreasing < highest_maxcg ) % I've seen Inf,Inf,...,Inf grab this condition
        % In this case, CG iterations were continually decreasing but never decreased below the last HF iteration
        % objective function value.  Don't count this a bork.
        if display_level > 1
            disp('CG iterations awere constantly decreasing, but not less than objective function at last HF iter.');
            disp('Starting from last CG solution with same data.');
        end
        
        cg_min_idx = NaN;
        
        objfun_constant_decreasing = objfun_cg;
        
        %disp('Experimental: decreasing lambda!!!');
        %lambda = rho_drop_val * lambda;

        % You might think that, with the all_simdata structures being passed around, not resampling the data on every hf
        % iteration is massive oversight.  This is only a problem if the eval_* calls modify simdata, generating a
        % history across hf iterations, and not a problem if the gen_data* function calls do it.  So far this hasn't
        % turned out to be the case, and only the gen_data* calls have modified simdata.  This is something we should
        % keep an eye on in the future, because the function interface, as it exists now, could definitely cause a
        % problem.  For example, the gradient computation computes some bit of state, that is expected by the gen_data*
        % routines, and then gets out of sync with the state as expected by the data generation functions.
	%
	% Still going to change it just to be safe, though this could cause a pathological hf update condition, which
        % is why the case was here in the first place.      -DCS:2013/04/24
        do_recompute_rho = false;
        do_recompute_gradient = false;
        do_resample_data = false;  % Why should we resample the data, if we just need to compute more iteartions?
        %do_recompute_gradient = false;
        %do_resample_data = false;
        
        % leave lambda alone, though see experimental above.
        if ~any(isnan(all_pn{end}))
            pn_cgstart = all_pn{end};
        else
            pn_cgstart = zeros(size(net.theta));
        end
        %maxcg = lowest_maxcg;
        
        maxcg = ceil(maxcg * 2.0);
        cgtol = cgtol / 2.0;
        if display_level > 1
            disp('Experimental: increasing maxcg and decreasing cgtol!!!');
        end
        
        %total_hf_suck_count = total_hf_suck_count + 1;   % Still count this as a fail.
        %total_hf_consecutive_suck_count = total_hf_consecutive_suck_count + 1;  % Still count this as a fail.
        ncgiters_constant_decreasing = ncgiters_constant_decreasing + maxcg;
    else
        if ncgiters_constant_decreasing >= highest_maxcg
            disp(['Failed because ncgiters_constant_decreasing < ' num2str(highest_maxcg) '.']);
        end
        if display_level > 1
            disp('Last CG evaluaation wasn''t good enough.  Trying to increase lambda.  Reseting CG start. Resampling data.');
        end
        total_hf_consecutive_suck_count  = total_hf_consecutive_suck_count + 1;
        total_hf_suck_count = total_hf_suck_count + 1;
        cg_min_idx = NaN;
        
        do_recompute_rho = false;  % How does this interface with trading out data?  DCS 05/04/2012
        do_recompute_gradient = true;
        do_resample_data = true;
        
        objfun_constant_decreasing = realmax;
        
        lambda = rho_boost_val * lambda;
        pn_cgstart = zeros(size(net.theta)); % reset the solution as well.
        maxcg = lowest_maxcg;
        ncgiters_constant_decreasing = 0;
    end
    
    % No matter what the previous logic, the max number of CG iterations has hard limits.
    if ( maxcg < lowest_maxcg )
        maxcg = lowest_maxcg;
    elseif ( maxcg > highest_maxcg )
        maxcg = highest_maxcg;
    end
    
    % Keep track of the number of consecutive validation error increases,
    % as it's a potential stopping condition.
    if ( objfun_test < min_test_objfun && cg_found_better_solution ) % Save only those solutions we'd keep anyways.
        total_consecutive_test_increase_count = 0;
        
        min_test_objfun = objfun_test;
        best_test_net = net;
        best_test_net_hf_iter = hf_iter;
        best_test_net_objfun_train = objfun_train;
        best_test_net_objfun_test = objfun_test;
    end
    if ( objfun_test > min_test_objfun && cg_found_better_solution )  % Can't use else because 'equals' means a failed HF iter, in all probability.
        total_consecutive_test_increase_count = total_consecutive_test_increase_count+1;
    end
    
    
    % Rho computation, phi(x) = 0.5 x'Ax - b'x, computed in the CG iterations.  This is used to compute lambda, a
    % crucial parameter used for computing the trust region of the quadratic approximation.
    if ( do_recompute_rho && hf_iter > 1 )
        
        % 	For rho, your denominator should just be the precise quadratic
        % minimized by CG.  You can compute it cheaply within cg from just b, x
        % and r (see my implementation which is now available on my website for
        % the exact formula).  The numerator can be computed on a larger dataset
        % (or the whole thing) but I often find that it is more effective on the
        % applications to have the numerator computed on the same subset of data
        % I use to compute the Gv products, as this lets lambda be reduced more
        % aggressively, which is sometimes what you want.  You should try doing
        % it both ways.
        
        % phi = 0.5*x'*Ax - b'*x;
        cg_phi = all_cg_phis(cgbt_min_idx);
        rho_numer = objfun - objfun_last;
        rho_denom = cg_phi;
        rho = rho_numer / rho_denom;
        
        % Sometimes rho will go negative.  At first i thought it was good to increase
        % lambda but now I don't think so.  If CG can't find a better solution,
        % lambda will increase above.
        if ( rho > 0.0 )
            if ( rho < rho_boost_thresh )
                lambda = rho_boost_val * lambda;
            elseif ( rho > rho_drop_thresh )
                lambda = rho_drop_val * lambda;
            end
        end
        if ( lambda ~= 0.0 )		% safety check if not identically equal to zero
            if ( lambda < lowest_lambda )
                lambda = lowest_lambda;
            end
            %elseif ( lambda > highest_lambda )
            %    lambda = highest_lambda;
            %end
        end
        if display_level > 1 
            disp(['CG Phi: ' num2str(cg_phi) ', rho: ' num2str(rho), ', lambda: ' num2str(lambda,6)]);
        end
    else
        if display_level > 1
            disp(['Lambda: ' num2str(lambda,6)]);
        end
    end
    
    % Display
    if display_level > 0
        if ( do_validation )
            disp(['    Objective function (train): ' num2str(objfun_train), ', Objective function (test): ', num2str(objfun_test)]);
        else
            disp(['    Objective function (train): ' num2str(objfun_train), '.']);
        end
    end
    if ( cg_found_better_solution )
        if ( hf_iter > 1 )
            if display_level > 0
                disp(['    (' cgbt_objfun ') Objective function - Objective function last: ', num2str(objfun - objfun_last)]);
            end
        end
    end
    
    
    %% Stopping conditions.

    % Stopping conditoin on the gradient magnitude.
    if ( grad_norm < tolfun )
        stop_string = 'Stopping because magnitude of gradient is less than the tolfun tolerance.';
        exit_flag = 1;
        go = 0;
    end
    

    % How is this different than looking at the magnitude of the gradient? x for fminunc is my theta.
%     if ( tolx_condition )
%         stop_string = ['Stopping because change in x was smaller than the tolx tolerance.'];
%         error_flag = 2;
%         go = 0;
%     end

    % Absolute stopping condition.  Note I don't have a relative stopping condition for the HF iterations.
    if ( abs(objfun - objfun_last) < objfun_tol )
        stop_string = ['Stopping because difference in objective function fell below: ' num2str(objfun_tol)];
        exit_flag = 3;  % fminunc: Change in the objective function value was less than the TolFun tolerance.
        go = 0;
    end
    
    if ( objfun < objfun_min )
        stop_string = ['Stopping because objective function fell below: ' num2str(objfun_min)];
        exit_flag = -3;  % fminunc: Objective function at current iteration went below ObjectiveLimit.
        go = 0;
    end
    
    % Reached maximum number of total CG failures.
    if ( total_hf_suck_count > hf_max_bork_count )
        stop_string = ['Stopping because total number of HF iteration failures is greater than ' num2str(hf_max_bork_count) '.'];
        exit_flag = -10;  % No fminunc equivalent.
        go = 0;
    end
    % Reached maximumum number of consecutive CG failures.
    if ( total_hf_consecutive_suck_count >= hf_max_consecutive_bork_count ) % not a good solution, check to see if we should stop.
        if display_level > 1 
            disp(['Rejecting last CG eval (iter: ' num2str(last_cgbt_eval_iter) '), since it''s more than the last CG pass.']);
        end
        stop_string = ['Stopping because HF failed ' num2str(hf_max_consecutive_bork_count) ' times in a row.'];
        exit_flag = -11;  % No fminunc equivalent.
        go = 0;
    end
    % Number of maximum consecutive increases in testing (validation) error
    % is reached.
    if ( total_consecutive_test_increase_count > hf_max_consecutive_test_iter_increases )
        stop_string = ['Stopping because the number of consecutive increases in validation error is ' num2str(total_consecutive_test_increase_count) '.'];
        exit_flag = -12; % No fminunc equivalent.
        go = 0;
    end
    if ( lambda >= max_lambda )
        stop_string = ['Stopping because the lambda is greater than the max lambda, ' num2str(lambda) '>' num2str(max_lambda) '.'];
        exit_flag = 10;  % Positive because I think this is a sign that we've reached a minimum.
        go = 0;
    end
    
    if do_time_hf_iters
        hf_iter_time = toc;
        total_time = total_time + hf_iter_time;
        if display_level > 2
            disp(['Elapsed time: ' num2str(hf_iter_time) ' seconds.']);
        end
    end
    
    
        
    % Optional plotting
    plot_stats = [];
    if do_plot && ~isempty(optional_plot_fun)
        plot_stats = optional_plot_fun(net, simparams, funs, cg_found_better_solution, f3, random_trial_idxs, forward_pass_T, forward_pass_s, ...
            v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
            m_targettrain_T, m_targettrain_s, m_targettest_t, all_optional_args, all_simdata, all_plot_stats);
    end
    
    
    
    % Save statistics
    all_grad_norms = [all_grad_norms grad_norm];
    all_lambdas = [all_lambdas lambda]; %#ok<*AGROW>
    all_rhos = [all_rhos rho];
    all_objfun_trains = [all_objfun_trains objfun_train];    
    all_objfun_trains_allobjs = [ all_objfun_trains_allobjs objfun_train_allobjs(:)];
    all_objfun_tests = [all_objfun_tests objfun_test];
    all_objfun_tests_allobjs = [all_objfun_tests_allobjs objfun_test_allobjs(:)];
    all_hf_iter_times = [all_hf_iter_times hf_iter_time];
    all_cg_iters_taken = [all_cg_iters_taken cg_min_idx]; % can be NaN for iters that failed.
    all_cg_iters_computed = [all_cg_iters_computed cg_gamma_idxs(end)];
    all_cg_found_better_solutions = [all_cg_found_better_solutions cg_found_better_solution];
    if do_save_thetas
        all_thetas(:, hf_iter) = net.theta(:);
    end
    if  ~isempty(plot_stats)
        all_plot_stats = [all_plot_stats plot_stats];   % must be column vector.
    end
    
    % Stats
    stats = [];
    stats.trainFileName = train_file_name;
    stats.trainFileContents = train_file_contents;
    stats.minibatchSize = S;
    stats.nExamples = T;
    stats.maxHFIters = max_hf_iters;
    stats.maxConsecutiveCGFailures = hf_max_consecutive_bork_count;
    stats.maxCGFailures = hf_max_bork_count;
    stats.nHFIters = hf_iter;
    stats.totalTime = total_time;
    stats.HFIterTimes = all_hf_iter_times;
    stats.solutionDidImprove = all_cg_found_better_solutions;
    stats.objFunTrain = all_objfun_trains;
    stats.objFunTrainAllObjs = all_objfun_trains_allobjs;
    stats.objFunTest = all_objfun_tests;
    stats.objFunTestAllObjs = all_objfun_tests_allobjs;
    stats.gradNorm = all_grad_norms;
    stats.maxLambda = max_lambda;
    stats.lambda = all_lambdas;
    stats.rho = all_rhos;
    stats.initLambda = init_lambda;
    stats.highestMaxCG = highest_maxcg;
    stats.lowestMaxCG = lowest_maxcg;
    stats.HFAbsTol = objfun_tol;
    stats.CGTol = cgtol;
    stats.CGBackTrackingGamma = gamma;
    stats.CGIncreaseFactor = cg_increase_factor;
    stats.CGItersTaken = all_cg_iters_taken;
    stats.CGItersComputed = all_cg_iters_computed;
    stats.lastThetaDecayFactor = last_w_decay_factor;
    stats.doGradOnFullData = do_grad_on_full_data;
    %stats.CGIncreaseThresholdPercentage = cg_increase_thresh_percentage;
    %stats.CGDecreaseThresholdPercentage = cg_decrease_thresh_percentage;
    %stats.CGIncreasePercentage = cg_increase_percentage;
    %stats.CGDecreasePercentage = cg_decrease_percentage;
    stats.rhoDropThreshold = rho_drop_thresh;
    stats.rhoBoostThreshold = rho_boost_thresh;
    stats.rhoDropValue = rho_drop_val;
    stats.rhoBoostValue = rho_boost_val;
    stats.doParallelNetwork = do_parallel_network;
    stats.doWrappersNetwork = do_wrappers_network;
    stats.doParallelObjFun = do_parallel_objfun;
    stats.doWrappersObjFun = do_wrappers_objfun;
    stats.doParallelGradient = do_parallel_gradient;
    stats.doWrappersGradient = do_wrappers_gradient;
    stats.doParallelCGAFun = do_parallel_cg_afun;
    stats.doWrappersCGAFun = do_wrappers_cg_afun;
    stats.savePath = save_path;
    stats.fileNamePart = filename_part;
    stats.pApMostNeg = smallest_pAp;
    stats.stopString = stop_string;
    stats.exitFlag = exit_flag;
    stats.plotStats = all_plot_stats;
    stats.thetas = all_thetas;    
    

    piter = piter + 1;
    if ( do_plot )
        plot_n_iters_back = 50;
        piters = piter-plot_n_iters_back:1:piter;
        piters = piters(piters>0);
        figure(f1); plot(piters, all_objfun_trains(piters), 'bo', 'linewidth', 4); hold on; axis tight;
        if do_validation
            figure(f1); plot(piters, all_objfun_tests(piters), 'rx', 'linewidth', 4); axis tight;
        end
        hold off;
        
        if do_plot_all_objectives
            figure(f2);
            subplot (1,2,1);
            for pidx = piters
                plot(pidx, all_objfun_trains_allobjs(:,pidx), 'o', 'linewidth', 4); hold on; axis tight;
                if do_validation
                    plot(pidx, all_objfun_tests_allobjs(:,pidx), 'x', 'linewidth', 4); axis tight;
                end
            end
            hold off;
            subplot( 1, 2, 2);
            aotra_diff = diff(all_objfun_trains_allobjs(:,piters),1,2);
            if do_validation
                aotea_diff = diff(all_objfun_tests_allobjs(:,piters),1,2);
            end
            ppidx = 0;
            for pidx = piters(2:end)
                ppidx = ppidx + 1;
                plot(pidx, aotra_diff(:,ppidx), 'o', 'linewidth', 4); hold on; axis tight;
                if do_validation
                    plot(pidx, aotra_diff(:,ppidx), 'x', 'linewidth', 4); hold on; axis tight;
                end
            end
            
            hold off;
        end
        
        pause(0.25);
    end
    % Save
    if ( mod(hf_iter, save_every) == 0 && ~isempty(filename_part) )
        save([save_path '/hfopt_' filename_part '_' num2str(hf_iter) '_' num2str(objfun_train) '_' num2str(objfun_test) ...
            '.mat'], 'net', 'stats', 'simparams', 'all_simdata');
    end
    
    %clear forward_pass_T;
    %clear forward_pass_s;
    %clear grad b;
    %clear all_pn;
    
end


stats.exitFlag = exit_flag;


% If the save string is non-null always save the network at the end of the optimization.
if ( ~isempty(filename_part) )
    save([save_path '/hfopt_' filename_part '_' num2str(hf_iter) '_' num2str(objfun_train) '_' num2str(objfun_test) ...
        '.mat'], 'net', 'stats', 'simparams', 'all_simdata');
end


% Condition where people cared about the validtion error.
%%% Note that value of NET changes here!
% This conditional wasn't saving the best hf iter, if the max count was
% acheived.  That's incorrect.  
%if ( total_consecutive_test_increase_count >= hf_max_consecutive_test_iter_increases )
if ( best_test_net_objfun_test < objfun_test )
    net = best_test_net;    % Note this makes it as the output parameter vector of the function.
    objfun_train = best_test_net_objfun_train;
    objfun_test = best_test_net_objfun_test;
    stats.lowestValidationErrorHFIter = best_test_net_hf_iter;  % Decision is to keep stats that went all the way to the last iter.
    save([save_path '/hfopt_' filename_part '_' num2str(best_test_net_hf_iter) '_' num2str(best_test_net_objfun_train) '_' num2str(best_test_net_objfun_test) ...
        '_lve.mat'], 'net', 'stats', 'simparams', 'all_simdata');
end

disp(stop_string);

% Outputs
theta = net.theta;

end



