function [digits, labels, digits_t, labels_t] = load_mnist_data(data_path, bias)

I = 785;				% 784 + 1 bias
Ttrain = 60000;				% number of examples
Ttest = 10000;

load ~/sandbox/forceproj/mlp_digit/data/digits_testing_C.txt;
load ~/sandbox/forceproj/mlp_digit/data/labels_testing_C.txt;

digits = bias*ones(I,Ttest);			% bias idx is 1 (or 0 in C).
digits(2:end,:) = digits_testing_C;

labeled_lines = labels_testing_C;
[ignore, labeled_lines_sorted] = sort(labeled_lines, 'descend');
labels = labeled_lines_sorted(1,:);   

load ~/sandbox/forceproj/mlp_digit/data/digits_training_C_1.txt;
load ~/sandbox/forceproj/mlp_digit/data/labels_training_C_1.txt;
load ~/sandbox/forceproj/mlp_digit/data/digits_training_C_2.txt;
load ~/sandbox/forceproj/mlp_digit/data/labels_training_C_2.txt;
load ~/sandbox/forceproj/mlp_digit/data/digits_training_C_3.txt;
load ~/sandbox/forceproj/mlp_digit/data/labels_training_C_3.txt;
load ~/sandbox/forceproj/mlp_digit/data/digits_training_C_4.txt;
load ~/sandbox/forceproj/mlp_digit/data/labels_training_C_4.txt;
load ~/sandbox/forceproj/mlp_digit/data/digits_training_C_5.txt;
load ~/sandbox/forceproj/mlp_digit/data/labels_training_C_5.txt;
load ~/sandbox/forceproj/mlp_digit/data/digits_training_C_6.txt;
load ~/sandbox/forceproj/mlp_digit/data/labels_training_C_6.txt;
disp(['Done loading.']);

digits_t = bias*ones(I,Ttrain);	% bias idx is 1 (or 0 in C).
digits_t(2:end,:) = [digits_training_C_1 digits_training_C_2 digits_training_C_3 digits_training_C_4 digits_training_C_5 digits_training_C_6 ];

labeled_lines_t = [labels_training_C_1 labels_training_C_2 labels_training_C_3 labels_training_C_4 labels_training_C_5 labels_training_C_6];
[ignore, labeled_lines_t_sorted] = sort(labeled_lines_t, 'descend');
labels_t = labeled_lines_t_sorted(1,:);   
