% train_pathological_cases2.m 
% Here are some examples of how to use the hfopt-matlab system.  
% These the examples from the original Schmidhuber Hochreiter papers. 

%% PREAMBLE, Parallel
matlabpool('local')

%% PREAMBLE, Paths
clear;

toplevel_path = '/Users/sussillo/Desktop/hfopt-matlab/';
pathos_path = [toplevel_path 'examples/pathologicals/matlab/'];
save_path = [toplevel_path 'examples/pathologicals/networks/']
addpath(genpath(toplevel_path));

cd(pathos_path)

rng('shuffle');  % seed the random generator based on the current time
seedStruct = rng;  % returns the current settings of the RNG

%%  Problem Type

pathos_type = 'addition'
%pathos_type = 'multiplication'
%pathos_type = 'xor';
pathos_subtype = '';

%pathos_type = 'delay';
%pathos_subtype = 'noiseless_memorization';

%pathos_type = 'temporalorder';
%pathos_subtype = '3bit';
switch pathos_type
    case {'xor', 'addition', 'multiplication'}    
        pathos_length = 100;
        nexamples_train = 2000;
        nexamples_test = 0;
        turnover_percentage = 1.0;   % HF breaks when the training data is modified on a per-iteration basis.
        minibatch_size = round(nexamples_train / 5);			% careful!  Is this #data or #trials?
    case {'delay'}
        switch pathos_subtype
            case 'noiseless_memorization'     
                pathos_length = 90;
                simparams.sequenceLength = 10;
                simparams.nIntegers = 5;
                nexamples_train = 10000;  % 5^10 is just under 10 million numbers!
                minibatch_size = 12*80;
                nexamples_test = 0;                
                turnover_percentage = 1.0;   % HF breaks when the training data is modified on a per-iteration basis.                
            case 'line'
                assert(false, 'Not implemented yet.');
            otherwise
                assert(false, 'Not implemented yet.');
        end
        
    case {'temporalorder'}
        switch pathos_subtype
            case {'2bit', '3bit'}
                do_orth = 1;
                g = 1.3;
                pathos_length = 200;
                nexamples_train = 10000;
                nexamples_test = 0;
                turnover_percentage = 1.0;   % HF breaks when the training data is modified on a per-iteration basis.
                minibatch_size = 12*80;			% careful!  Is this #data or #trials?
            otherwise
                assert(false, 'Not implemented yet.');
        end
        
    otherwise
        assert(false, 'Not implemented yet.');
end


simparams.pathosType = pathos_type;
simparams.pathosSubtype = pathos_subtype;
simparams.pathosLength = pathos_length;

objective_function = 'sum-of-squares';
% Network related stuff

net_type = 'rnn-trials';
switch pathos_type
    case {'xor', 'addition', 'multiplication'}
        layer_types = {'linear', 'tanh', 'linear' };
        layer_sizes = [ 2 100 100 1];
        do_orth = true;
        do_matrix_exponential = false;
        burn_length = 0;
        numconn = 15;
        g = 1.4; 
        %  disp('Experimental: Structural damping is off!');
        mu = 0.03;				
        lambda = 0.001;
    case {'delay'}
        switch pathos_subtype
            case 'line'
                net_type = 'rnn-trials';
                layer_types = {'linear', 'tanh', 'linear' };
                layer_sizes = [ 1 100 100 1];
                burn_length = 0;
                numconn = 15;
                mu = 0.03;				
                lambda = 1e0;
            case 'noiseless_memorization'
                net_type = 'rnn-trials';
                objective_function = 'cross-entropy';
                layer_types = {'linear', 'tanh', 'softmax' };
                %layer_sizes = [ 4 100 100 4];
                layer_sizes = [ 7 100 100 7];
                g = 1.3;
                do_orth = false;
                burn_length = 0;
                numconn = 15;
                mu = 0.03;				
                lambda = 1e0;
        end
    case {'temporalorder'}
        switch pathos_subtype
            case '2bit'
                net_type = 'rnn-trials';
                layer_types = {'linear', 'tanh', 'linear' };
                layer_sizes = [ 6 100 100 4];		% 2 bit
                burn_length = 0;
                numconn = 15;
                mu = 0.03;				
                lambda = 1e0;
            case '3bit'
                net_type = 'rnn-trials';
                layer_types = {'linear', 'tanh', 'linear' };
                layer_sizes = [ 6 100 100 8];		% 2 bit
                burn_length = 0;
                numconn = 15;
                mu = 0.03;				
                lambda = 1e0;
            otherwise
                assert(false, 'Case not implemented yet.');
        end
    otherwise
        disp('Case not implemented yet.');
end

%cd sandbox/forceproj/hfopt/pathologicals/matlab; train_pathological_cases



%% Network stuff

% Parameter scaling by layer.  This is based on g * randn(N,N)/sqrt(N), for
% example.
g_by_layer = zeros(3,1);  % Should be size 3, (I->N) (N->N) (N->M)
g_by_layer(1) = 1.0;
g_by_layer(2) = g;
g_by_layer(3) = 1.0; 

% Discrete system as specified by dt = tau.
tau = 1.0;
dt = 1.0;

% Cost mask by layer, multiplies the weight cost 
cm_fac_by_layer = ones(3,1);   % Should be size 3, (I->N) (N->N) (N->M)
cm_fac_by_layer(1) = 1.0;
cm_fac_by_layer(2) = 1.0;
cm_fac_by_layer(3) = 1.0;  % See if we can make the state bigger


do_init_state_biases_random = false;  
do_learn_biases = true;
do_init_state_init_random = false;
do_learn_state_init = true;


%%  Create / Load Network
do_new_net = 1;
do_load_old_net = 0;
do_load_old_net_from_file = 0;
%do_matrix_exponential = 0;

filename_part = [pathos_type pathos_subtype];

if ( do_new_net )
    disp('Initializing new network.');
    net = init_rnn(layer_sizes, layer_types, g_by_layer, objective_function, ...
        'doinitstatebiasesrandom', do_init_state_biases_random, ...
        'dolearnbiases', do_learn_biases, ...
        'doinitstateinitrandom', do_init_state_init_random, ...
        'dolearnstateinit', do_learn_state_init, ...
        'cmfacbylayer', cm_fac_by_layer, 'numconn',  numconn, 'tau', tau, 'dt', dt, 'mu', mu);
    orignet2 = net;
    
    if ( do_matrix_exponential )
        disp('Trying matrix exponential.');
        [n_Wru_v, n_Wrr_n, m_Wzr_n, n_r0_1, n_br_1, m_bz_1] = unpackRNN(net, net.theta);
        
        n_Wrr_exp_n = expm(n_Wrr_n - eye(size(n_Wrr_n)));
        net.theta = packRNN(net, n_Wru_v, n_Wrr_exp_n, m_Wzr_n, n_r0_1, n_br_1, m_bz_1);
    end
    if ( do_orth )
        disp('Trying orthogonal matrix.');
        [n_Wru_v, ~, m_Wzr_n, n_r0_1, n_br_1, m_bz_1] = unpackRNN(net, net.theta);       
        N = layer_sizes(2);
        n_Wrr_n = randn(N,N);
        n_Wrr_orth_n = g_by_layer(2)*orth(n_Wrr_n);
        net.theta = packRNN(net, n_Wru_v, n_Wrr_orth_n, m_Wzr_n, n_r0_1, n_br_1, m_bz_1);
    end
end

net.layers(2).initLength = burn_length;		% burn at the beginning

%%

weight_cost = 0e-6;
simparams.weightCost = weight_cost;
simparams.burnLength = 0;

inputs_last = {};
targets_last = {};

switch pathos_type
    case {'xor', 'addition', 'multiplication'}
        
        gen_pathos_data_train2 = @(net, inputs_last, targets_last, simparams, all_simdata, do_inputs, do_targets) gen_pathos_data2(net, inputs_last, targets_last, simparams, nexamples_train, all_simdata, do_inputs, do_targets);            
        gen_pathos_data_test2 = @(net, inputs, targets, simparams, all_simdata, do_inputs, do_targets) gen_pathos_data2(net, inputs_last, targets_last, simparams, nexamples_test, all_simdata, do_inputs, do_targets);            
        
        optional_plot_fun = @pathos_optional_plot_fun2;
        
    case {'delay'}
        assert(false, 'Need to update functions to second version.');
        gen_pathos_data_train = @(net, inputs_last, targets_last, simparams) gen_delay_line(net, simparams, nexamples_train, turnover_percentage, ...
            inputs_last, targets_last);
        gen_pathos_data_test = @(net, inputs_last, targets_last, simparams) gen_delay_line(net, simparams, nexamples_test, turnover_percentage, ...
            inputs_last, targets_last);
        optional_plot_fun = @delay_optional_plot_fun;
        
    case {'temporalorder'}
        assert (false, 'Need to update functions to second version.');
        gen_pathos_data_train = @(net, inputs_last, targets_last, simparams) gen_temporal_order_data(net, simparams, nexamples_train, turnover_percentage, ...
            inputs_last, targets_last);
        gen_pathos_data_test = @(net, inputs_last, targets_last, simparams) gen_temporal_order_data(net, simparams, nexamples_test, turnover_percentage, ...
            inputs_last, targets_last);
        optional_plot_fun = @temporalorder_optional_plot_fun;
        
    otherwise
        disp('Case not implemented yet.');
end


%%  Optimization settings and optimization call

save_every = 20;
save_name = ['pathos_' simparams.pathosType '_T' num2str(simparams.pathosLength)];
if ~isempty(simparams.pathosSubtype)
    save_name = [save_name '_' simparams.pathosSubtype];
end

% This seems to be important because the network needs to get into its state.


max_hf_iters = 1000;

% Training related stuff.
objfuntol = 1e-16;
maxcgiter = 60;
maxconsecutivetestincreases = 100;
mincgiter = 10;
cgepsilon = 1e-7;

max_consec_failures = 100;
do_parallel_network = true;
do_parallel_objfun = true;
do_parallel_gradient = true;
do_parallel_cg_afun = true;

TRAINING_IDX = 1;
VALIDATION_IDX = 2;
all_simdata{TRAINING_IDX}(1:nexamples_train) = struct;
all_simdata{VALIDATION_IDX}(1:nexamples_test) = struct;

frob_norm = 2e-6;
net.frobeniusNormRecRecRegularizer = frob_norm % was 1e-5

[opttheta, objfun_train, objfun_test, stats] = hfopt2(net, gen_pathos_data_train2, [], gen_pathos_data_test2, [], ...
    'maxhfiters', max_hf_iters, ...
    'S', minibatch_size, 'doplot', 1, 'objfuntol', objfuntol, ...
    'initlambda', lambda, 'weightcost', weight_cost, ...
    'highestmaxcg', maxcgiter, 'lowestmaxcg', mincgiter, ...
    'maxconsecutivetestincreases', maxconsecutivetestincreases,...
    'cgtol', cgepsilon, 'nettype', net_type, ...
    'optplotfun', optional_plot_fun, ..., %'optevalfun', optional_eval_fun, ...
    'samplestyle', 'random_blocks', ... 
    'simdata', all_simdata, ...
    'savepath', save_path, 'filenamepart', filename_part, 'saveevery', save_every, 'paramstosave', simparams, 'maxcgfailures', 1000, ...
    'doparallelnetwork', do_parallel_network, ...
    'doparallelobjfun', do_parallel_objfun, ...
    'doparallelgradient', do_parallel_gradient, ...
    'doparallelcgafun', do_parallel_cg_afun);

origtheta = net.theta;


net.theta = opttheta;

%lambda = stats.lambda(end);
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_r0_1, n_br_1, m_bz_1] = unpackRNN(net, net.theta);

