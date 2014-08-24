%% PREAMBLE, Parallel
matlabpool('local')

%% PREAMBLE, Paths
cd ~/sandbox/forceproj/omri/fixed_points/matlab
addpath('~/sandbox/worlddomination_forceproj/trunk/howitworks/matlab/');
addpath('~/sandbox/worlddomination_forceproj/trunk/hfopt/matlab/');



rng('shuffle');  % seed the random generator based on the current time
seedStruct = rng;  % returns the current settings of the RNG


%%  Load Data

% There are some basic network parameters necessary to know how to make
% the data.
dt = 0.1;
tau = 1.0;

ntrials = 5;
%ntrials = 24;


ninputs = 1;

time = 800;
times = dt:dt:time;
ntimes = length(times);

inputs = cell(1,ntrials);
targets = cell(1,ntrials);
targets1 = cell(1,ntrials);
freq_rads = zeros(1,ntrials);
burn_length = round(ntimes / 2.0);  
for i = 1:ntrials
    
    freq_rad = 0.05*i/ntrials + 0.1;
    freq_rads(i) = freq_rad;
    
    inputs{i} = 1.0 * sin(freq_rad * times) + (i/ntrials +0.25)* ones(1,ntimes);
    inputs{i}(burn_length+1:end) = (i/ntrials +0.25)* ones(1,burn_length);
    
    targets1{i} = sin(freq_rad * times);
    targets{i} = NaN(ntrials, ntimes);
    targets{i}(i,:) = targets1{i};           
end
targets = fliplr(targets);  % Faster stuff nearer to zero.



%% Create Network Parameters
do_abort_if_save_path_exists = false;
save_name = 'sine_gen';
save_path_part = '/Users/sussillo/sandbox/forceproj/omri/data/networks/hfopt/sinegen_1x_postpub/';

%n_ind_params = 1;
%as = 1;
%gs = 0.95;

n_ind_params = 1;
%gs = linspace(0.9, 1.5, n_ind_params)
gs = 1.5;
%gs = 1;
as = 1;
gsm = gs;
asm = repmat(as, n_ind_params, 1)

nparams = length(gs)
rpidxs = randperm(nparams)


%% Train the Networks
do_debug = false;
do_one_output = true;

close all;

allnets = {};
for i = 1:nparams
    
    idx = rpidxs(i);
    
    g = gsm(idx);
    alpha = asm(idx);

    disp(['Training model with g: ' num2str(g) ' and alpha: ' num2str(alpha) '.']);
    
    % Create the directories, if they don't already eixst.   
    save_path_end_dir = [save_name '_g' num2str(g,3) '_a' num2str(alpha,3)];
    save_path = [save_path_part save_path_end_dir];
    if ( ~exist(save_path, 'dir') )
        mkdir(save_path);
    elseif do_abort_if_save_path_exists
        disp(['Model with these parameters already exists... continuing!']);
        continue;
    end
        
    % Network related stuff.
    net_type = 'rnn-trials';
    ninputs = 1;
    N = 200;   %Defined above for dynamical noise.
    if do_one_output  % Harder with one output.
        M = 1;
    else
        M = ntrials;
    end
    layer_sizes = [ ninputs N N M];
    layer_types = {'linear', 'tanh', 'linear' };
    objective_function = 'sum-of-squares';
    numconn = 15;    
    network_noise = 0.0;  % don't forget multiplication by sqrt(dt)
    
    g_by_layer = zeros(3,1);  % Should be size 3, (I->N) (N->N) (N->M)   
    g_by_layer(1) = 1.0;
    g_by_layer(2) = g;    
    g_by_layer(3) = 0.0;

    save_every = 5;             % With the gigantic noise matrices, each save is about 0.3GB.
    mu = 0.03;				

    lambda = 0.0002;           % lambda and minibatch size are intimately related.
    minibatch_size = ntrials;		% careful!  Is this #data or #trials?  Bigger is probably better here cuz of data load times.

    % These are the two main stopping criteria.  Useful to set well, if
    % there's a bunch of sims
    max_hf_iters = 500;			
    max_hf_failures = 500;
    max_consec_test_increase = 10;
    objfuntol = 1e-9;
    frob_norm = 1e-3;
    
    
    % Training related stuff.
    do_learn_biases = 1;
    do_init_state_biases_random = 0;
    do_learn_state_init = 1;
    do_init_state_init_random = 0;  % zero is probably good init for oscillatory behavior

    maxcgiter = 100;
    mincgiter = 10;
    cgepsilon = 1e-6;
    
    weight_cost = 0e-3;  
    cm_fac_by_layer = ones(3,1);
    cm_fac_by_layer(1) = 0.0; 
    cm_fac_by_layer(2) = 0.0; 
    cm_fac_by_layer(3) = 1.0;  % See if we can make the state bigger


    disp('Initializing new network.');
    net = init_rnn(layer_sizes, layer_types, g_by_layer, objective_function, ...
		   'cmfacbylayer', cm_fac_by_layer, 'numconn',  numconn, 'tau', tau, 'dt', dt, 'netnoisesigma', network_noise, 'mu', mu, 'transfunparams', alpha, ...
           'dolearnbiases', do_learn_biases, 'doinitstatebiasesrandom', do_init_state_biases_random, ...
           'dolearnstateinit', do_learn_state_init, 'doinitstateinitrandom', do_init_state_init_random);
    net.layers(2).initLength = burn_length;		% burn at the beginning
    
    net.frobeniusNormRecRecRegularizer = frob_norm;
    
    do_plot = 1;  % Somehow after running a bunch of these, I beleive the figures are screwing up matlab so everything comes to a screaming halt.
    
    simparams.seedStruct = seedStruct;
    simparams.doNewNet = true;
    simparams.doLoadOldNet = false;
    simparams.doLoadOldNetFromFile = false;
    simparams.oldNetLoadPath = '';
    simparams.oldNetPackageName = '';
    simparams.doResetToOriginalTheta = false;
    simparams.doMatrixExponential = false;
    simparams.trainingDataString = 'simulated';
    simparams.time = time;
    simparams.g = g;
    simparams.alpha = alpha;
    simparams.nTotalTrials = ntrials;
    simparams.freqRads = freq_rads;
    simparams.doOneOutput = do_one_output;
    
    % Define functions
    optional_plot_fun = @sinegen_plotfun_postpub;   
    
    if ~do_debug
        do_parallel_network = true;
        do_parallel_objfun = true;
        do_parallel_gradient = true;
        do_parallel_cg_afun = true;
    else
        do_parallel_network = false;
        do_parallel_objfun = false;
        do_parallel_gradient = false;
        do_parallel_cg_afun = false;
    end
     
    
    % Now train the network to do the task.  
    if do_one_output
        targets_opt = targets1;
    else
        targets_opt = targets;
    end
    [opttheta, objfun_train, objfun_test, stats] = hfopt2(net, inputs, targets_opt, {}, {}, ...
						  'maxhfiters', max_hf_iters, 'maxcgfailures', max_hf_failures, 'maxconsecutivetestincreases', max_consec_test_increase, ...
						  'S', minibatch_size, 'doplot', do_plot, 'objfuntol', objfuntol, ...
						  'initlambda', lambda, 'weightcost', weight_cost, ...
						  'highestmaxcg', maxcgiter, 'lowestmaxcg', mincgiter, ...
						  'cgtol', cgepsilon, 'nettype', net_type, ...
						  'optplotfun', optional_plot_fun, ...
                          'samplestyle', 'random_rows', ...
						  'savepath', save_path, 'filenamepart', save_name, 'saveevery', save_every, ...
                          'paramstosave', simparams, ...
                          'doparallelnetwork', do_parallel_network, ...
                          'doparallelobjfun', do_parallel_objfun, ...
                          'doparallelgradient', do_parallel_gradient, ...
                          'doparallelcgafun', do_parallel_cg_afun);



    net.theta = opttheta;
    allnets{idx} = net;        
end 