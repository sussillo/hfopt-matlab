%% Preamable
matlabpool('local') 

%% train_mnist_classifier_with_hf

addpath('~/sandbox/worlddomination_forceproj/trunk/howitworks/matlab/');
addpath('~/sandbox/worlddomination_forceproj/trunk/hfopt/matlab/');
addpath('/Users/sussillo/work/sandbox/worlddomination_forceproj/trunk/mlp_digit/matlab');


toplevel_path = '/Users/sussillo/Desktop/hfopt-matlab/';
pathos_path = [toplevel_path 'examples/mnist/matlab/'];
save_path = [toplevel_path 'examples/mnist/networks/']
addpath(genpath(toplevel_path));

cd(pathos_path)


%% Load the data

data_path = [toplevel_path 'examples/mnist/data/'];
bias = 0.0;
[digits, labels, digits_t, labels_t] = load_mnist_data(data_path, bias);

digits(end,:) = [];
digits_t(end,:) = [];

% Training related stuff.
ndigits_train = 60000;
ndigits_test = 10000;

labels_hotone = full(sparse(labels,1:ndigits_test,1));
labels_t_hotone = full(sparse(labels_t,1:ndigits_train,1));




%%  Set up the network parameters

net_type = 'db';
layer_sizes = [ 784 500 500 2000 10];
%layer_sizes = [ 784 500 10];
layer_types = {'rectlinear' 'rectlinear' 'rectlinear' 'softmax'};
%layer_types = {'rectlinear' 'softmax'};


%objective_function = 'sum-of-squares';
objective_function = 'cross-entropy';
numconn = Inf;
g = 1.0;
g_by_layer = g*ones(length(layer_sizes),1);
g_by_layer(end) = 1.0;

cm_fac_by_layer = ones(length(layer_sizes),1);

do_new_net = 1;
if do_new_net
    disp('Initializing new network.');
    net = init_dn(layer_sizes, layer_types, g_by_layer, objective_function, ...
        'wcfacbylayer', cm_fac_by_layer, 'numconn',  numconn);
    orignet = net;
else
    disp('Loading pre-trained network.');
    package_path = '/Users/sussillo/sandbox/forceproj/mlp_digit/data/networks/';
    package = load([package_path 'hfopt_mnist_classify_5_0.21398_0.21319.mat']);
    net = package.net;
end
net.dropOutPercentageInput = 0.2;
net.dropOutPercentageHidden = 0.0;


weight_cost = 2e-5;
objfuntol = 1e-9;
maxcgiter = 150;
mincgiter = 10;
cgepsilon = 1e-8;
optional_plot_fun = @mnist_optional_plot_fun;



%% Train with HF  (normal)

close all;
save_every = 25;
save_name = 'mnist_classify';

simparams.init = 1;
% This seems to be important because the network needs to get into its state.

lambda = 1e-2;
cgbt_objfun = 'train';			% try test objective function with examples swapping out.
max_hf_iters = 10000;

minibatch_size = 10000/4;
% Training related stuff.
objfuntol = 1e-16;
maxcgiter = 150;
mincgiter = 10;
cgepsilon = 1e-7;

max_consec_failures = 100;
do_parallel_network = false;
do_parallel_objfun = false;
do_parallel_gradient = false;
do_parallel_cg_afun = false;
do_parallel_precon = false;

net_type = 'dn';  % deep net
origtheta = net.theta;
[opttheta, objfun_train, objfun_test, stats] = hfopt2(net, digits_t, labels_t_hotone, digits, labels_hotone, ...
    'maxhfiters', max_hf_iters, 'cgbtobjfun', cgbt_objfun, ...
    'S', minibatch_size, 'doplot', 1, 'objfuntol', objfuntol, ...
    'initlambda', lambda, 'weightcost', weight_cost, ...
    'highestmaxcg', maxcgiter, 'lowestmaxcg', mincgiter, ...
    'maxhffailures', 10000, ...
    'maxconsecutivecgfailures', max_consec_failures, ...
    'cgtol', cgepsilon, 'nettype', net_type,  ...
    'optplotfun', optional_plot_fun, 'samplestyle', 'random_rows', ...
    'paramstosave', simparams, ...
    'savepath', save_path, 'filenamepart', save_name, 'saveevery', save_every, ...
    'doparallelnetwork', do_parallel_network, ...
    'doparallelobjfun', do_parallel_objfun, ...
    'doparallelgradient', do_parallel_gradient, ...
    'doparallelcgafun', do_parallel_cg_afun, ...
    'doparallelprecon', do_parallel_precon);


net.theta = opttheta;

