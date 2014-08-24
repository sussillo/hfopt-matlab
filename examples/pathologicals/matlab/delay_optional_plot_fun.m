function plot_stats = delay_optional_plot_fun(net, simparams, fun, did_objfun_improve_this_iter, f, trial_idxs, forward_pass_s, ...
    v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
    m_targettrain_T, m_targettrain_s, m_targettest_t, optional_args)

plot_stats = [];

if isempty(forward_pass_s)
    return;
end

figure(f);
hold off;

%forward_pass_s = eval_network(net, v_inputtrain_s); % Not content to use mismatch from before CG iter and network update.
% xxx don't forget that the network has the r0 in it.
T = size(forward_pass_s{1}{3},2);

for i = 1:length(forward_pass_s)
    if ( i > 4 )
        continue;
    end
    subplot(4,3,3*(i-1)+1);
    %plot(v_inputtrain_s{i}(1,:), '-kx', 'linewidth', 1)
    %hold on;
    imagesc(m_targettrain_s{i});   
    
    subplot(4,3,3*(i-1)+2);
    imagesc(forward_pass_s{i}{3});
    
    subplot(4,3,3*(i-1)+3);
    plot(forward_pass_s{i}{1}(1:5,:)', 'b');
    axis tight;
    
end

