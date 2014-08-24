%% PREAMBLE, Parallel
matlabpool

%% PREAMBLE, Paths
cd ~/sandbox/forceproj/hfopt/pathologicals/matlab
addpath('~/sandbox/worlddomination_forceproj/trunk/howitworks/matlab/');
addpath('~/sandbox/worlddomination_forceproj/trunk/hfopt/matlab/');
addpath('~/sandbox/forceproj/hfopt/pathologicals/matlab/');


rng('shuffle');  % seed the random generator based on the current time
seedStruct = rng;  % returns the current settings of the RNG

%%  Load the Model

save_path = '~/Dropbox/sandbox/forceproj/hfopt/pathologicals/networks/';
%save_path = '~/sandbox/forceproj/hfopt/pathologicals/networks/';

%net_name = 'hfopt_pathos_addition_T200_290_3.1333e-06_Inf.mat';
net_name = 'hfopt_pathos_multiplication_T200_235_8.0581e-06_Inf.mat';


wrapper = load([save_path net_name]);
net = wrapper.net;
simparams = wrapper.simparams;


inv_trans_fun = net.layers(2).invTransFun;
trans_fun = net.layers(2).transFun;

[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_br_1, m_bz_1] = unpackRNN(net, net.theta);
[n_OWru_v, n_OWrr_n, m_OWzr_n, n_Ox0_c, n_Obr_1, m_Obz_1] = unpackRNN(net, net.originalTheta);

I = net.layers(1).nPre;
N = net.layers(2).nPre;
B = net.layers(3).nPost;

winwout = [n_Wru_v m_Wzr_n'];
winwout_normed = normify(winwout);


%weight_cost = 0e-10;		% this parameter sholud also be tied to either the network or another saved structure.

weight_cost = 0e-6;  % Should be stored in simparams.

eval_network_rnn = create_eval_network_rnn(weight_cost);
eval_objfun_rnn = create_eval_objfun_rnn(weight_cost);
eval_gradient_rnn = create_eval_gradient_rnn(weight_cost);
eval_cg_afun_rnn = create_eval_cg_afun_rnn(weight_cost);
eval_preconditioner_rnn = [];
eval_network = create_eval_network(eval_network_rnn, weight_cost);
eval_objfun = create_eval_objfun(eval_objfun_rnn, weight_cost);
eval_gradient = create_eval_gradient(eval_gradient_rnn, weight_cost);
eval_cg_afun = create_eval_cg_afun(eval_cg_afun_rnn, weight_cost);
eval_preconditioner = [];
funs.evalNetwork = eval_network;
funs.evalObjfun = eval_objfun;
funs.evalGradient = eval_gradient;
funs.evalCGAFun = eval_cg_afun;
funs.evalPreconditioner = eval_preconditioner;


eval_preconditioner = [];


%%  Load the data.

nexamples = 1000;
turnover_percentage = 0.0;

[inputs, targets, ~] = gen_pathos_data(net, simparams, nexamples, turnover_percentage, {}, {});


%% Evaluate tons of trials

TvV_T = 1;
TvV_V = 2;

forward_pass = eval_network(net, inputs, targets, TvV_V, 1:nexamples, {});
objfun = eval_objfun(net, inputs, targets, TvV_V, 1:nexamples, {});

disp(['Objective function evaluation: ' num2str(objfun) '.']);

Racts = {};
Xacts = {};
netouts = {};
for i = 1:length(forward_pass)
    Racts{i} = forward_pass{i}{1};
    Xacts{i} = inv_trans_fun(Racts{i});
    netouts{i} = forward_pass{i}{3};
end


%% Plot a couple examples.

f = figure;
pathos_optional_plot_fun(net, simparams, funs, false, f, false, forward_pass, {}, inputs, {}, {}, targets, {}, {});



%% Create a single average example.

add_vals = [0.9 0.1];
add_idxs = [8, 29];

%xxx needs to have multiple ending times.

navg = 1000;
avg_inputs = {};
avg_target = targets{1};  % irrelevant
avg_target(end) = mean(add_vals);
for i = 1:navg
    
    avg_inputs{i} = zeros(2,100);
    avg_inputs{i}(1,:) = rand(1,100);
    avg_inputs{i}(1,add_idxs) = add_vals;
    avg_inputs{i}(2,add_idxs) = 1;
    avg_targets{i} = avg_target;
end

avg_forward_pass = eval_network(net, avg_inputs, avg_targets, TvV_T, 1:navg, {});

avg_traj = [];
for i = 1:navg
    if i > 1
        avg_traj = avg_traj + avg_forward_pass{i}{1};
    else
        avg_traj = avg_forward_pass{i}{1};
    end
end
avg_traj = avg_traj / navg;
x_example_traj = inv_trans_fun(avg_forward_pass{1}{1});
x_avg_traj = inv_trans_fun(avg_traj);

%% PCA

R = cell2mat(Racts);
meanR = mean(R,2);
Rz = bsxfun(@minus, R, meanR);

Cr = (Rz*Rz')/size(Rz,2);
[V,Dr] = eig(Cr);
dr = diag(Dr);
[dr, sidxs_v] = sort(dr, 'descend');
dr(dr < eps) = eps;

Rpca.C = Cr;			% it's sorted, motherstopper!
Rpca.V = V(:,sidxs_v);
Rpca.d = dr;
Rpca.sortIdxs = 1:N;
Rpca.mean = meanR;


X = net.layers(2).invTransFun(R);

meanRS = mean(X,2);
Xz = bsxfun(@minus, X, meanRS);
Cx = (Xz*Xz')/size(Xz,2);
[V,Dx] = eig(Cx);
dx = diag(Dx);
[dx, sidxs_u] = sort(dx, 'descend');
dx(dx < eps) = eps;

RSpca.C = Cx;			% it's sorted, motherstopper!
RSpca.V = V(:,sidxs_u);
RSpca.d = dx;
RSpca.sortIdxs = 1:N;
RSpca.mean = meanRS;


line_width = 2;
font_size = 18;
eigs_to_show = 30;
figure;
plot(1:eigs_to_show, log10(RSpca.d(1:eigs_to_show)), '-kx', 'linewidth', line_width);
hold on;
plot(1:eigs_to_show, log10(Rpca.d(1:eigs_to_show)), '-rx', 'linewidth', line_width);
%axis tight;
xlabel('\lambda #', 'fontsize', font_size);
ylabel('log10(\lambda)', 'fontsize', font_size);
legend('X eigenvalues', 'R eigenvalues');
set(gca(gcf), 'fontsize', font_size);

%ds = 10;
%P = downsample((Xpca.U'*Xz')', ds)';
%Q = downsample((Rpca.V'*Rz')', ds)';

frac_cumsum_x = cumsum(RSpca.d) / sum(RSpca.d);
frac_cumsum_r = cumsum(Rpca.d) / sum(Rpca.d);



var_explained = 0.99
thing = find(frac_cumsum_x > var_explained);
pc_num_var_explained_x = thing(1)
thing = find(frac_cumsum_r > var_explained);
pc_num_var_explained_r = thing(1)


%% zero inputs and pulses

ninputs = 2;
zero_input_length = 10;
zero_input{1} = zeros(ninputs, zero_input_length);
zero_output{1} = zeros(1,zero_input_length);

const_input = zero_input;
label_val = 0.0;
const_input_vals = [0.5; label_val];  % Open question as to whether the "constant" input should be the input mean.
const_input{1}(1,:) = const_input_vals(1);
const_input{1}(2,:) = const_input_vals(2);


%% Find fixed points


nfps = 200;
init_eps = 0.01;
fun_tol = 8e-8;
[fp_struct_avg, fpd] = find_many_fixed(net, nfps, X, init_eps, 100.0, fun_tol, 'constinput', const_input_vals);

% order the fixed points
fp_end = -m_Wzr_n';
mnorms = [];
for i = 1:nfps
    fp = fp_struct_avg(i).FP;
    mnorms(i) = norm(fp - fp_end);
end
[~,sidxs] = sort(mnorms);
fp_struct_avg = fp_struct_avg(sidxs);

fpnet = net;
for i = 1:nfps
    fpnet.theta = packRNN(net, n_Wru_v, n_Wrr_n, m_Wzr_n, fp_struct_avg(i).FP, n_br_1, m_bz_1);
    forward_pass_fps{i} = eval_network(fpnet, const_input, zero_output, TvV_T, 1, {}, 'doparallel', false);
    Z_fp{i} = forward_pass_fps{i}{1}{3};
    X_fp{i} = inv_trans_fun(forward_pass_fps{i}{1}{1});
end

figure;
for i = 1:nfps
    plot(Z_fp{i}, 'r');
    hold on;
end

%% FP lines

label_val = 0.0;
ninputs = 2;
zero_input_length = 10;
zero_input{1} = zeros(ninputs, zero_input_length);
zero_output{1} = zeros(1,zero_input_length);

nlines = 25;
const_input_values = linspace(0,1,nlines);

nfp_lines = zeros(1,nlines);
fp_struct_lines = cell(1,nlines);
for i =1:nlines
    
    const_input = zero_input;
    const_input_vals = [const_input_values(i); label_val];  % Open question as to whether the "constant" input should be the input mean.
    const_input{1}(1,:) = const_input_vals(1);
    const_input{1}(2,:) = const_input_vals(2);
   
    nfps = 100;
    init_eps = 0.01;
    fun_tol = 1e-7;  % worked for addition
    %fun_tol = 1e-6;
    [fp_struct_lines{i}, fpd] = find_many_fixed(net, nfps, X, init_eps, 100.0, fun_tol, 'constinput', const_input_vals);
    fp_struct_lines{i} = fp_struct_lines{i}(find([fp_struct_lines{i}.FPNorm] < 1e30));
    nfps_lines(i) = length(fp_struct_lines{i});

    
    % order the fixed points
    fp_end = -m_Wzr_n';
    mnorms = [];
    for j = 1:nfps_lines(i)
        fp = fp_struct_lines{i}(j).FP;
        mnorms(j) = norm(fp - fp_end);
    end
    [~,sidxs] = sort(mnorms);
    fp_struct_lines{i} = fp_struct_lines{i}(sidxs);   
    
 end

%%  3D Plot

do_check = true;  %  check the values of xla0, xla1, xi1, and xi2 from analysis below.
do_plot_axis = 1;
do_fp_zeros = 1;
do_plot_fps = 1;
do_plot_rights = 1;
do_plot_lefts = 1;

Vno = winwout_normed;
projMean = zeros(N,1);
[V,~] = qr(Vno, 0);
dims = [1 2 3];

Xacts_zm = {};
for i = 1:length(Xacts)
    Xacts_zm{i} = bsxfun(@minus, Xacts{i}, projMean);
end

%dists = dist(fps);
%[i,j] = max(dists);
%unique(j)

axis_vecs = V'*bsxfun(@minus, winwout_normed, projMean);
rvec_colors = [0 0 0; 0 0 1; 1 0 0];

figure;

if do_plot_axis
    for i = 1:3
        plot3([0 axis_vecs(dims(1),i)], [0 axis_vecs(dims(2),i)], [0 axis_vecs(dims(3),i)], 'color', rvec_colors(i,:), 'linewidth', 2)
        hold on;
    end
end


if do_plot_fps
    dims = [1 2 3];
    %plot3(fps_proj(dims(1),:), fps_proj(dims(2),:), fps_proj(dims(3),:), 'rx', 'linewidth', 5)
    hold on;
    nfpspoints = 10;
    
    
    for k = 1:nlines
        nevs = 1;
        for i = 1:nfps_lines(k)
            if mod(i,2) == 0
                continue;
            end
            for j = 1:nevs
                rvec = real(fp_struct_lines{k}(i).eigenVectors(:,j));
                slow_mode_color = 'r';
                line_width = 4;
                
                ev_proj =  V'*rvec;
                fps = real([fp_struct_lines{k}.FP]);  % Finds slightly imaginary shit.
                fps_proj = V'*bsxfun(@minus, fps, projMean);

                fp_proj = fps_proj(:,i);
                plot3([fp_proj(dims(1))],[fp_proj(dims(2))],[fp_proj(dims(3)) ], 'rx', 'Color', slow_mode_color, 'linewidth', line_width)
                if do_plot_rights
                    plot3([fp_proj(dims(1)) fp_proj(dims(1))+ev_proj(dims(1))],[fp_proj(dims(2)) fp_proj(dims(2))+ev_proj(dims(2))],[fp_proj(dims(3)) fp_proj(dims(3))+ev_proj(dims(3))], 'Color', slow_mode_color)
                    plot3([fp_proj(dims(1)) fp_proj(dims(1))-ev_proj(dims(1))],[fp_proj(dims(2)) fp_proj(dims(2))-ev_proj(dims(2))],[fp_proj(dims(3)) fp_proj(dims(3))-ev_proj(dims(3))], 'Color', slow_mode_color)
                end
                if ( do_plot_lefts )
                    line_width = 2;
                    lvec = real(fp_struct_lines{k}(i).leftEigenVectors(j,:)');
                    ev_proj =  V'*lvec;
                    plot3([fp_proj(dims(1)) fp_proj(dims(1))+ev_proj(dims(1))],[fp_proj(dims(2)) fp_proj(dims(2))+ev_proj(dims(2))],[fp_proj(dims(3)) fp_proj(dims(3))+ev_proj(dims(3))], 'Color', 'g', 'linewidth', line_width)
                end
            end
        end
    end
end

nplots = 50;
cm1 = winter(100);
cm2 = autumn(100);
sidx = 14;
for i = 1:nplots
    output = targets{i}(end);
    netout = netouts{i}(end);
    rproj = V'*Xacts_zm{i};
    
    input_idxs = find(inputs{i}(2,:));
    plot_idxs = input_idxs; % r0 + see effect one step after
    plot_idxs = [plot_idxs length(inputs{i})];
    
    input1 = inputs{i}(1, input_idxs(1));
    input2 = inputs{i}(1, input_idxs(2));
    
    cmidx_out = ceil(output*100);
    cmidx_in1 = ceil(input1*100);
    cmidx_in2 = ceil(input2*100);
    
    %plot3(rproj(dims(1), plot_idxs(1)), rproj(dims(2),plot_idxs(1)), rproj(dims(3),plot_idxs(1)), 'o', 'Color', cm1(cmidx_in1, :), 'markersize', 10, 'linewidth', 3)
    %text(rproj(dims(1), plot_idxs(1)), rproj(dims(2),plot_idxs(1)), rproj(dims(3),plot_idxs(1)), num2str(input1,2));
    
    %plot3(rproj(dims(1), plot_idxs(2)), rproj(dims(2),plot_idxs(2)), rproj(dims(3),plot_idxs(2)), 'o', 'Color', cm1(cmidx_in2, :), 'markersize', 10, 'linewidth', 3)
    %text(rproj(dims(1), plot_idxs(2)), rproj(dims(2),plot_idxs(2)), rproj(dims(3),plot_idxs(2)), num2str(input2,2));
    
    plot3(rproj(dims(1), plot_idxs(3)), rproj(dims(2),plot_idxs(3)), rproj(dims(3),plot_idxs(3)), 'v', 'Color', cm2(cmidx_out, :), 'markersize', 10, 'linewidth', 3)
    %text(rproj(dims(1),plot_idxs(3)), rproj(dims(2),plot_idxs(3)), rproj(dims(3),plot_idxs(3)), [num2str(input1,2) 'x' num2str(input2,2) '=' num2str(output,2)]);
    %text(rproj(dims(1),plot_idxs(3)), rproj(dims(2),plot_idxs(3)), rproj(dims(3),plot_idxs(3)), [num2str(output,2)]);
    
    %plot3(rproj(dims(1), plot_idxs), rproj(dims(2),plot_idxs), rproj(dims(3),plot_idxs), '.', 'Color', cm(cmidx, :), 'markersize', 10)
    
end

rproj = V'*bsxfun(@minus, x_avg_traj, projMean);
plot3(rproj(dims(1), :), rproj(dims(2),:), rproj(dims(3),:), 'Color', 'k', 'linewidth', 2)

rproj = V'*bsxfun(@minus, x_example_traj, projMean);
plot3(rproj(dims(1), :), rproj(dims(2),:), rproj(dims(3),:), 'Color', [0.5 0.5 0.5], 'linewidth', 1)


do_check = false;
if do_check
    rprojs = V'*bsxfun(@minus, xla0(:,1:nplots), projMean);
    plot3(rprojs(dims(1), :), rprojs(dims(2),:), rprojs(dims(3),:), 'x', 'Color', 'c', 'linewidth', 2)
    
    rprojs = V'*bsxfun(@minus, xla1(:,1:nplots), projMean);
    plot3(rprojs(dims(1), :), rprojs(dims(2),:), rprojs(dims(3),:), 'x', 'Color', 'm', 'linewidth', 2)
end

%axis equal;
axis normal;

%% Eigenvalue plots

do_eigenvalue_plot = 1;

markersize = 20;

figure;
x = linspace(-1,1,200);
y = sqrt(1-x.^2);
plot(x, y); hold on; plot(x, -y)
hold on;
plot([fp_struct_avg(50).eigenValues]', 'r.', 'markersize', markersize);


figure;
plot(x, y); hold on; plot(x, -y)
hold on;
colors = jet(nfps);
for i = 1:nfps
    plot([fp_struct_avg(i).eigenValues]', '.', 'color', colors(i,:), 'markersize', markersize);
end

axis equal;



%% Left eigenvector analysis with traj points

%close all;
% Get the points n steps after the indicator onsets.

xla0 = zeros(N,nexamples);  % Meant to be on line attractor right before 1st impulse.
xla1 = zeros(N,nexamples);  % Meant to be on line attractor right before 2nd impulse.

xi1 = zeros(N,nexamples);    % Meant to be off line attractor, after 1st impulse has affected system.
xi2 = zeros(N,nexamples);    % Meant to be off line attractor, after 2nd impulse has affected system.

% Values to average. 
as = zeros(1,nexamples);  % First input value.
bs = zeros(1,nexamples);  % Second input value.
apres = zeros(1,nexamples);
bpres = zeros(1,nexamples);
targs = zeros(1,nexamples);  % Average value

v1s = zeros(1,nexamples);  % Network output, right before second input.

% Get the readout values for all the fixed points.
fp_ros = zeros(1,nfps);
for i = 1:nfps
    fp_ros(i) = m_Wzr_n * trans_fun(fp_struct_avg(i).FP) + m_bz_1;    
end

npast = 0;

% Find the fixed point index for the reaodut value for the value just before t1 (around 0.65).
[~, fpros_idx] = min((fp_ros-0.65).^2);
mina_idxs = zeros(nexamples,1);
minb_idxs = zeros(nexamples,1);
minapre_idxs = zeros(nexamples,1);
minbpre_idxs = zeros(nexamples,1);
for i = 1:nexamples
    tidxs = find(inputs{i}(2,:));    
    tidx1 = tidxs(1) + npast;
    tidx2 = tidxs(2) + npast;
    tidxpre1 = tidx1-1;
    if tidxpre1 == 0
        tidxpre1 = 1;
    end
    tidxpre2 = tidx2-1;
            
    a = inputs{i}(1,tidx1);
    b = inputs{i}(1,tidx2);    
    
    if tidx1 > 1 
        apre = inputs{i}(1,tidxpre1);
    else
        apre = a;
    end
    bpre = inputs{i}(1,tidxpre2);        
    
    as(i) = a;    
    bs(i) = b;    
    
        
    % These are nice to collect, but I don't think they matter.  It's where the x point is before these values come in.
    [~, mina_idxs(i)] = min((a - const_input_values).^2);
    [~, minb_idxs(i)] = min((b - const_input_values).^2);
    [~, minapre_idxs(i)] = min((apre - const_input_values).^2);
    [~, minbpre_idxs(i)] = min((bpre - const_input_values).^2);

%     mina_idxs(i) = (nlines-1)/2;   % This is just using the average one.
%     minb_idxs(i) = (nlines-1)/2;
%     minapre_idxs(i) = (nlines-1)/2;   % This is just using the average one.
%     minbpre_idxs(i) = (nlines-1)/2;
    
    if tidxs(1) > 1
        xla0(:,i) = Xacts{i}(:,tidxpre1);
    else
        xla0(:,i) = fp_struct_avg(fpros_idx).FP;
    end            
    xla1(:,i) = Xacts{i}(:,tidxpre2);  
       
    xi1(:,i) = Xacts{i}(:,tidx1);
    xi2(:,i) = Xacts{i}(:,tidx2);
    
    targs(i) = targets{i}(end);
    
    v1s(i) = netouts{i}(tidxpre2);  % Try to get the value that's stable, just before the input comes in.   
end

% Find the linear systems that are closest to the x state on the line attractor just before the input comes in.
fpidxs_xla0 = zeros(1,nexamples);
fpidxs_xla1 = zeros(1,nexamples);


% all_fps = [];
% for i = 1:nlines
%    all_fps = [all_fps [fp_struct_lines{i}.FP]];
% end

% Don't want dot products, want distances!
% for i = 1:nexamples
%     lidxa = minapre_idxs(i);
%     fp_mat = [fp_struct_lines{lidxa}.FP];
%     [~, midx] = min(sqrt(sum((repmat(xla0(:,i), 1, nfps_lines(lidxa)) - fp_mat).^2, 1)));   
%     fpidxs_xla0(i) = midx;
%     
%     lidxb = minbpre_idxs(i);
%     fp_mat = [fp_struct_lines{lidxb}.FP];   
%     [~, midx] = min(sqrt(sum((repmat(xla1(:,i), 1, nfps_lines(lidxb)) - fp_mat).^2, 1)));
%     fpidxs_xla1(i) = midx;
% end

for i = 1:nexamples
    lidxa = mina_idxs(i);
    fp_mat = [fp_struct_lines{lidxa}.FP];
    [~, midx] = min(sqrt(sum((repmat(xi1(:,i), 1, nfps_lines(lidxa)) - fp_mat).^2, 1)));   
    fpidxs_xla0(i) = midx;
    
    lidxb = minb_idxs(i);
    fp_mat = [fp_struct_lines{lidxb}.FP];   
    [~, midx] = min(sqrt(sum((repmat(xi2(:,i), 1, nfps_lines(lidxb)) - fp_mat).^2, 1)));
    fpidxs_xla1(i) = midx;
end


figure;
%scatter(targs, (as + bs)/2, 'k');
scatter(targs, (as.*bs), 'k');
hold on; 

crazyvals = zeros(1,nexamples);
for i = 1:nexamples
    fpidx0 = fpidxs_xla0(i);
    fpidx1 = fpidxs_xla1(i);
    fp1 = fp_struct_lines{mina_idxs(i)}(fpidx0).FP;
    fp2 = fp_struct_lines{minb_idxs(i)}(fpidx0).FP;
    l1 = real(fp_struct_lines{mina_idxs(i)}(fpidx0).leftEigenVectors(1,:)');
    l2 = real(fp_struct_lines{minb_idxs(i)}(fpidx1).leftEigenVectors(1,:)');
    r1 = real(fp_struct_lines{mina_idxs(i)}(fpidx0).eigenVectors(:,1));
    r2 = real(fp_struct_lines{minb_idxs(i)}(fpidx1).eigenVectors(:,1));
    
    % The nonlinearity assumption asserts itself in how the the inputs end up being represented.  I'm hoping the dynamics are nevertheless linear.
    %v0 = fp_struct_avg(fpros_idx).FP; 
    s = 1;  % I think the left eigenvectors can end up backwards.
    v0 = trans_fun(xla0(:,i));  % This captures the first value 0.3666 or so that's already on the line attractor.
    v1 = (v0 + s*r1*(l1'*(trans_fun(xi1(:,i)))));  % First input comes in.        
    v2 = (v1 + s*r2*(l2'*(trans_fun(xi2(:,i)))));  % Second input comes in.

    
    %v0 = trans_fun(xla0(:,i));  % This captures the first value 0.3666 or so that's already on the line attractor.
    %v1 = v0 + (s*r1*(l1'*(xi1(:,i)-fp1))+fp1);  % First input comes in.        
    %v2 = v1 + (s*r2*(l2'*(xi2(:,i)-fp2))+fp2);  % Second input comes in.

    
    crazyvals(i) = m_Wzr_n * trans_fun(v2) + m_bz_1;
end


scatter(targs, crazyvals, 5, [1 0 0]);

%xlim([0 1])
%ylim([0 1])

%% Show the projection of the first state onto the output vector.

figure; plot(m_Wzr_n * trans_fun(xla0) + m_bz_1, 'x')

%% Check out nonlinear mapping of inputs. 

% Uses xi1 and xi2 from above.
xstates = [xi1 xi2];
rstates = trans_fun(xstates);

meanRS = mean(rstates,2);
Xz = bsxfun(@minus, rstates, meanRS);
Cx = (Xz*Xz')/size(Xz,2);
[V,Dx] = eig(Cx);
dx = diag(Dx);
[dx, sidxs_u] = sort(dx, 'descend');
dx(dx < eps) = eps;

RSpca.C = Cx;			% it's sorted, motherstopper!
RSpca.V = V(:,sidxs_u);
RSpca.d = dx;
RSpca.sortIdxs = 1:N;
RSpca.mean = meanRS;

beta = regress([as bs]', [rstates' ones(2*nexamples,1)]);

% Rows correspond to obvervations
npcas = 3;
mappedX = (RSpca.V(:,1:npcas)'*rstates)';
Vax = RSpca.V(:,1:npcas)'*[winwout beta(1:end-1)];

figure;
veclabels = {'V', 'I', 'R', '\beta'};
dims = [1 2 3];
colors = [0 0 0; 0 0 1; 1 0 0; 0.5 0.5 0.5];
for i = 1:size(Vax,2)
    plot3([0 Vax(dims(1),i)], [0 Vax(dims(2),i)], [0 Vax(dims(3), i)], 'color', colors(i,:), 'linewidth', 2)
    hold on;
end
legend(veclabels);

scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3), 10, [as bs], 'filled');
hold on;

figure; plot(log10(abs((RSpca.d))), '-x')

grid off;
axis normal;






