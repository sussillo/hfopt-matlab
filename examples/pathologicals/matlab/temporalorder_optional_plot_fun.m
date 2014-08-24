function plot_stats = temporalorder_optional_plot_fun(net, simparams, fun, did_objfun_improve_this_iter, f, trial_idxs, forward_pass_s, ...
    v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
    m_targettrain_T, m_targettrain_s, m_targettest_t, optional_args)

plot_stats = [];


if isempty(forward_pass_s)
    return;
end

figure(f);
hold off;

%forward_pass_s = eval_network(net, v_inputtrain_s); % Not content to use mismatch from before CG iter and network update.
delay = 50;				% xxx hard coded for now.

% xxx don't forget that the network has the r0 in it.
T = size(forward_pass_s{1}{3},2);

for i = 1:length(forward_pass_s)
    if ( i > 4 )
        continue;
    end
    subplot(4,2,i);
    plot(forward_pass_s{i}{3}', '-xb');
    
    axis tight;
    ylim([0 1]);
    hold off;
    
    subplot(4,2,4+i);
    plot(forward_pass_s{i}{1}(1:5,:)', 'b');
    
end

