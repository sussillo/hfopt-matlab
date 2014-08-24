function optional_args = pm_maze_evalfun(~, simparams, ~, v_inputtrain_T, ~, ~, ~, ~)

% This function is allowed to know the intimate details of how the input is encoded.  The two context lines are 3 and
% 4 and if 3 is 1 then it's motion and if 4 is 1 then its a color trial.

% Determine which of two types of integration based on the input value.
optional_args = cell(1,2);
[~, T] = size(v_inputtrain_T);
for i = 1:T    % Note this is capital T, so it gets all trials, no subset minibatches.
    
    % This is used to set the initial conditions in the rnn_hf_allfun.m.  For PM, this should be one for each trial.
    %optional_args{1}(i).conditionID = i; % For each example, we can have a different initial condition.  Unfortunately named in this application.    
    
    optional_args{1}(i).conditionID = simparams.conditions(i); % For each example, we can have a different initial condition.  Unfortunately named in this application.    
end
optional_args{2} = [];  
