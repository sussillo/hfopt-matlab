%% PREAMBLE, Parallel
matlabpool 12

%% PREAMBLE, Paths
cd ~/sandbox/forceproj/hfopt/pathologicals/matlab
addpath('~/sandbox/worlddomination_forceproj/trunk/howitworks/matlab/');
addpath('~/sandbox/worlddomination_forceproj/trunk/hfopt/matlab/');
addpath('~/sandbox/forceproj/hfopt/pathologicals/matlab/');


rng('shuffle');  % seed the random generator based on the current time
seedStruct = rng;  % returns the current settings of the RNG

%%  Load the Model

save_path = '~/Dropbox/sandbox/forceproj/hfopt/pathologicals/networks/';
net_name = 'hfopt_pathos_delay_T90_noiseless_memorization_370_2.3544e-07_Inf.mat';

wrapper = load([save_path net_name]);
net = wrapper.net;
simparams = wrapper.simparams;

sequence_length = simparams.sequenceLength;
nintegers = simparams.nIntegers;

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

nexamples = nintegers^sequence_length;
turnover_percentage = 0.0;

[inputs, targets, ~] = gen_delay_line(net, simparams, nexamples, turnover_percentage, {}, {});
 
%for i = 1:32; subplot(1,2,1); imagesc(inputs{i}); subplot(122); imagesc(targets{i}); pause(0.25); end

%% Evaluate tons of trials

TvV_T = 1;
TvV_V = 2;

late_idxs = 90:100;

forward_pass = eval_network(net, inputs, targets, TvV_V, 1:nexamples, {});
objfun = eval_objfun(net, inputs, targets, TvV_V, 1:nexamples, {});

disp(['Objective function evaluation: ' num2str(objfun) '.']);

Racts = {};
Xacts = {};
netouts = {};
for i = 1:length(forward_pass)
    Racts{i} = forward_pass{i}{1};
    Xacts{i} = inv_trans_fun(Racts{i});
    XActsLate{i} = inv_trans_fun(Racts{i}(:,late_idxs));
    netouts{i} = forward_pass{i}{3};
end


%% Plot a couple examples.

f = figure;
delay_optional_plot_fun(net, simparams, funs, false, f, false, forward_pass, {}, inputs, {}, {}, targets, {}, {});

figure; imagesc(inputs{1})


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

Xpca.C = Cx;			% it's sorted, motherstopper!
Xpca.V = V(:,sidxs_u);
Xpca.d = dx;
Xpca.sortIdxs = 1:N;
Xpca.mean = meanRS;


line_width = 2;
font_size = 18;
eigs_to_show = 30;
figure;
plot(1:eigs_to_show, log10(Xpca.d(1:eigs_to_show)), '-kx', 'linewidth', line_width);
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

frac_cumsum_x = cumsum(Xpca.d) / sum(Xpca.d);
frac_cumsum_r = cumsum(Rpca.d) / sum(Rpca.d);



var_explained = 0.999
thing = find(frac_cumsum_x > var_explained);
pc_num_var_explained_x = thing(1)
thing = find(frac_cumsum_r > var_explained);
pc_num_var_explained_r = thing(1)


%% zero inputs and pulses

ninputs = I;
zero_input_length = 10;
zero_input{1} = zeros(ninputs, zero_input_length);
zero_output{1} = zeros(1,zero_input_length);

const_input = zero_input;
const_input_vals = [zeros(nintegers,1); 1; 0];  % Open question as to whether the "constant" input should be the input mean.
const_input{1}(nintegers+1,:) = const_input_vals(nintegers+1);


%% Find fixed points

do_ghost_kmeans = true;  % Useful if you know what you are looking for.
nfp_tries = 200;
init_eps = 0.01;
fun_tol = 1e-11;
[fp_struct, fpd] = find_many_fixed(net, nfp_tries, cell2mat(XActsLate), init_eps, 100.0, fun_tol, 'constinput', const_input_vals);

gnorms = [fp_struct.FPNorm];
gidxs = find(gnorms < 1e20);
fp_struct = fp_struct(gidxs);
nfps = length(fp_struct);

figure; stem([fp_struct.FPNorm]);

% 
if do_ghost_kmeans
    nfps = 3;  % Have to look for yourself.
    fps = [fp_struct.FP];  % This is a hack and one should really look
    [a,b] = kmeans(fps', nfps);
    new_fp_struct = fp_struct;
    new_fp_struct(1:end) = [];
    for i = 1:nfps
        aidxs = find(a == i);
        aidx = aidxs(1);
        new_fp_struct(i) = fp_struct(aidx);
    end
    fp_struct = new_fp_struct;
end

% Haven't mapped this function over yet, but I should.  DCS May 18, 2012
%closeness_tol = 1e-2;
%[fp_struct, ufp_idxs] = get_unique_fixed_points(net, fp_struct, closeness_tol, fun_tol);
    

fpnet = net;
for i = 1:nfps
    fpnet.theta = packRNN(net, n_Wru_v, n_Wrr_n, m_Wzr_n, fp_struct(i).FP, n_br_1, m_bz_1);
    forward_pass_fps{i} = eval_network(fpnet, const_input, zero_output, TvV_T, 1, {}, 'doparallel', false);
    Z_fp{i} = forward_pass_fps{i}{1}{3};
    X_fp{i} = inv_trans_fun(forward_pass_fps{i}{1}{1});
end

figure;
for i = 1:nfps
    plot(Z_fp{i}', 'r');
    hold on;
    plot(X_fp{i}(1:5,:)', 'b');
end

% These two fixed points are very similiar, as evidenced by their leading eigenvectors.
figure; 
plot(diag(abs(fp_struct(2).eigenVectors' * fp_struct(1).eigenVectors)),'-x');


%% Find ghosts

do_ghost_kmeans = true;  % Useful if you know what you are looking for.
nghost_tries = 200;
init_eps = 0.01;
fun_tol = 0.006;
[g_struct, fpd] = find_many_fixed(net, nghost_tries, cell2mat(XActsLate), init_eps, 100.0, fun_tol, 'constinput', const_input_vals);

gnorms = [g_struct.FPNorm];
gidxs = find(gnorms < 1e20);
g_struct = g_struct(gidxs);
nghosts = length(g_struct);

figure; stem([g_struct.FPNorm]);

% 
if do_ghost_kmeans
    nghosts = 10;  % Have to look for yourself.
    ghosts = [g_struct.FP];  % This is a hack and one should really look
    [a,b] = kmeans(ghosts', nghosts);
    new_g_struct = g_struct;
    new_g_struct(1:end) = [];
    for i = 1:nghosts
        aidxs = find(a == i);
        aidx = aidxs(1);
        new_g_struct(i) = g_struct(aidx);
    end
    g_struct = new_g_struct;
end
    

fpnet = net;
for i = 1:nghosts
    fpnet.theta = packRNN(net, n_Wru_v, n_Wrr_n, m_Wzr_n, g_struct(i).FP, n_br_1, m_bz_1);
    forward_pass_fps{i} = eval_network(fpnet, const_input, zero_output, TvV_T, 1, {}, 'doparallel', false);
    Z_fp{i} = forward_pass_fps{i}{1}{3};
    X_fp{i} = inv_trans_fun(forward_pass_fps{i}{1}{1});
end

figure;
for i = 1:nghosts
    plot(X_fp{i}(1:5,:)', 'b');
end


%% Hand select the two for the moment.

good_fp_idxs = [1 13];

fp_struct = fp_struct(good_fp_idxs);
nfps = length(fp_struct);

%% Plot the eigenvalues

x = linspace(-1,1,1000);
y = sqrt(1-x.^2);
for i = 1:nfps; 
    figure; 
    plot(fp_struct(i).eigenValues, 'x', 'linewidth', 3, 'markersize', 20)
    hold on;
    plot(x, y, 'k');
    plot(x,-y, 'k');
    
    npos = fp_struct(i).nPos;
    plot(real(fp_struct(i).eigenValues(1:npos)), imag(fp_struct(i).eigenValues(1:npos) ), 'rx', 'linewidth', 3, 'markersize', 20)
end


%%  3D Plot

%close all;

do_check = true;  %  check the values of xla0, xla1, xi1, and xi2 from analysis below.
do_plot_axis = 1;
do_fp_zeros = 1;
do_plot_fps = 1;
do_plot_ghosts = 0;
do_plot_lefts = false;

nplots = 32;


Vno = winwout(:, [4 5 6]);
projMean = zeros(N,1);
rvec_colors = [0 0 1; 1 0 0; 1 0 0];
axis_text = ['I3'; 'O1'; 'O2'];

%Vno = Xpca.V(:,1:3);
%projMean = Xpca.mean;
%axis_text = ['PC1'; 'PC2'; 'PC3'];
%rvec_colors = [0 0 0; 0 0 0; 0 0 0];


% ufpidx = 1;  % Pick the unstable one, by hand, after each fp finding.
% Vno = [real(fp_struct(ufpidx).eigenVectors(:,1)) imag(fp_struct(ufpidx).eigenVectors(:,1)) fp_struct(ufpidx).eigenVectors(:,3)];
% projMean = fp_struct(ufpidx).FP;
% rvec_colors = [0 0 1; 0 0 1; 1 0 0];
% axis_text = ['V1'; 'V2'; 'V3'];

[V,~] = qr(Vno, 0);
dims = [1 2 3];

Xacts_zm = {};
for i = 1:length(Xacts)
    Xacts_zm{i} = bsxfun(@minus, Xacts{i}, projMean);
end

%axis_vecs = V'*winwout_normed;
axis_vecs = V'*Vno;


figure;

if do_plot_axis
    for i = 1:3
        voffset = 0*[-4 -4 -4];
        scale = 3;
        plot3([voffset(1) voffset(1)+scale*axis_vecs(dims(1),i)], ...
            [voffset(2) voffset(2)+scale*axis_vecs(dims(2),i)], ...
            [voffset(3) voffset(3)+scale*axis_vecs(dims(3),i)], 'color', rvec_colors(i,:), 'linewidth', 2);
        text([voffset(1)+scale*axis_vecs(dims(1),i)], ...
            [voffset(2)+scale*axis_vecs(dims(2),i)], ...
            [voffset(3)+scale*axis_vecs(dims(3),i)], axis_text(i,:), 'fontsize', 18);
        hold on;
    end
end

for i = 1:nplots
    rproj = V'*bsxfun(@minus, Xacts{i}, projMean);
    plot3(rproj(dims(1), :), rproj(dims(2),:), rproj(dims(3),:), 'Color', 'k', 'linewidth', 1)
    plot3(rproj(dims(1), end), rproj(dims(2),end), rproj(dims(3),end), 'rx', 'linewidth', 3, 'markersize', 10)
    plot3(rproj(dims(1), 1), rproj(dims(2),1), rproj(dims(3),1), 'ro', 'linewidth', 3, 'markersize', 10)
end

if do_plot_fps
    dims = [1 2 3];
    %plot3(fps_proj(dims(1),:), fps_proj(dims(2),:), fps_proj(dims(3),:), 'rx', 'linewidth', 5)
    hold on;
    nfpspoints = 10;    
    
    nevs = 1;
    for i = 1:nfps
        for j = 1:nevs
            rvec = real(fp_struct(i).eigenVectors(:,j));
            slow_mode_color = 'r';
            
            ev_proj =  V'*rvec;
            fps = real([fp_struct.FP]);  % Finds slightly imaginary shit.
            fps_proj = V'*bsxfun(@minus, fps, projMean);
            
            fp_proj = fps_proj(:,i);
            plot3(fp_proj(dims(1)), fp_proj(dims(2)), fp_proj(dims(3)),'bo', 'markersize', 10, 'linewidth',4);
            plot3([fp_proj(dims(1)) fp_proj(dims(1))+ev_proj(dims(1))],[fp_proj(dims(2)) fp_proj(dims(2))+ev_proj(dims(2))],[fp_proj(dims(3)) fp_proj(dims(3))+ev_proj(dims(3))], 'Color', slow_mode_color)
            plot3([fp_proj(dims(1)) fp_proj(dims(1))-ev_proj(dims(1))],[fp_proj(dims(2)) fp_proj(dims(2))-ev_proj(dims(2))],[fp_proj(dims(3)) fp_proj(dims(3))-ev_proj(dims(3))], 'Color', slow_mode_color)
            if ( do_plot_lefts )
                line_width = 2;
                lvec = real(fp_struct(i).leftEigenVectors(j,:)');
                ev_proj =  V'*lvec;
                plot3([fp_proj(dims(1)) fp_proj(dims(1))+ev_proj(dims(1))],[fp_proj(dims(2)) fp_proj(dims(2))+ev_proj(dims(2))],[fp_proj(dims(3)) fp_proj(dims(3))+ev_proj(dims(3))], 'Color', 'g', 'linewidth', line_width)
            end
        end   
    end
end


if do_plot_ghosts
    dims = [1 2 3];
    %plot3(fps_proj(dims(1),:), fps_proj(dims(2),:), fps_proj(dims(3),:), 'rx', 'linewidth', 5)
    hold on;
    nfpspoints = 10;    
    
    nevs = 1;
    for i = 1:nghosts
        for j = 1:nevs
            fps = real([g_struct.FP]);  % Finds slightly imaginary shit.
            fps_proj = V'*bsxfun(@minus, fps, projMean);
            
            fp_proj = fps_proj(:,i);
            plot3(fp_proj(dims(1)), fp_proj(dims(2)), fp_proj(dims(3)),'mv', 'markersize', 10, 'linewidth',4);
        end   
    end
end


%axis equal;
axis normal;


%% Left Eigenvector analysis with input vectors.

readout_vals = zeros(nfps, 1);

alpha = 0.5;
V = [winwout alpha*n_Wru_v(:,1)+n_Wru_v(:,2)];
lvals = zeros(size(V,2),nfps);
for i = 1:nfps
    lz_evec = fp_struct(i).leftEigenVectors(1,:)';
    
    
    readout_vals(i) = (fp_struct(i).FP)'*m_Wzr_n' + 0*m_bz_1;
    lvals(:,i) = real(V'*lz_evec);    
end

figure; 
plot(readout_vals, lvals(1,:), '-kx');
hold on; 
plot(readout_vals, lvals(2,:), '-x');
plot(readout_vals, lvals(3,:), '-rx');
%plot(readout_vals, lvals(4,:), '-mx');
set(gca(gcf), 'fontsize', 18);
xlabel('Readout value');
ylabel('Projection of lefts');
legend('V', 'I', 'R', 'location', 'northwest');
axis tight;
xlim([0 1]);


%% Left eigenvector analysis with traj points
