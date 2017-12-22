%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EECS E6720: Homework 1
%
% Author: Michael Nguyen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
float('double');
%- Set up constants
a = 1;
b = 1;
%- Set up the labels
fileID = fopen('data/README');
xlabels = textscan(fileID, '%s');
xlabels = xlabels{1};
fclose(fileID);

%- Set up the training data
train_X = csvread('data/X_train.csv');
train_Y = csvread('data/label_train.csv');
train_index_1 = find(train_Y == 1);
train_index_0 = find(train_Y == 0);
N = length(train_Y);

prob_y1 = (1 + length(train_index_1)) / (N + 2);
prob_y0 = (1 + length(train_index_0)) / (N + 2);

%- Set up the testing data
test_X = csvread('data/X_test.csv');
test_Y = csvread('data/label_test.csv');
test_index_1 = find(test_Y == 1);
test_index_0 = find(test_Y == 0);

num_test = length(test_Y);
probs_test = zeros(num_test, 2);
label_test = zeros(num_test, 1);

for i = 1:num_test
	Xhat = test_X(i,:);
	probs_test(i, 1) = predictive_fcn(Xhat, train_X(train_index_1, :), a, b) * prob_y1;
	probs_test(i, 2) = predictive_fcn(Xhat, train_X(train_index_0, :), a, b) * prob_y0;

	probs_test(i,:) = probs_test(i, :) / sum( probs_test(i, :) );

	if(probs_test(i,1) > probs_test(i,2))
		label_test(i) = 1;
	end
end

accuracy = sum( label_test == test_Y ) / num_test;

cm_00 = sum( label_test(test_index_0) == test_Y(test_index_0) ) / length(test_index_0);
cm_11 = sum( label_test(test_index_1) == test_Y(test_index_1) ) / length(test_index_1);
cm_01 = sum( label_test(test_index_1) ~= test_Y(test_index_1) ) / length(test_index_1);
cm_10 = sum( label_test(test_index_0) ~= test_Y(test_index_0) ) / length(test_index_0);

cm_00 = sum( label_test(test_index_0) == test_Y(test_index_0) ) ;
cm_11 = sum( label_test(test_index_1) == test_Y(test_index_1) ) ;
cm_01 = sum( label_test(test_index_1) ~= test_Y(test_index_1) ) ;
cm_10 = sum( label_test(test_index_0) ~= test_Y(test_index_0) ) ;


lambda1 = (sum(train_X(train_index_1,:),1) + a) / (b + length(train_index_1));
lambda0 = (sum(train_X(train_index_0,:),1) + a) / (b + length(train_index_0));

missind = find( label_test ~= test_Y );
missind = missind( randperm(length(missind)) );

figure(1); clf;

subplot(3,1,1);
hold on;
plot(1:54, lambda0);
plot(1:54, lambda1);
stem(1:54, test_X(missind(1),:));
title(['Prediction probabilities (1, 0): ' num2str( probs_test(missind(1),:) ) ])
ylabel('Feature Value');
legend('\lambda_0', '\lambda_1', ['Truth label: ' num2str(test_Y(missind(1)))])
set(gca,'Xtick',1:54);
set(gca,'XtickLabel',xlabels);
xtickangle(60);
axis tight
hold off;

subplot(3,1,2);
hold on;
plot(1:54, lambda0);
plot(1:54, lambda1);
stem(1:54, test_X(missind(2),:));
title(['Prediction probabilities (1, 0): ' num2str( probs_test(missind(2),:) ) ])
ylabel('Feature Value');
legend('\lambda_0', '\lambda_1', ['Truth label: ' num2str(test_Y(missind(2)))])
set(gca,'Xtick',1:54);
set(gca,'XtickLabel',xlabels);
xtickangle(60);
axis tight
hold off;

subplot(3,1,3);
hold on;
plot(1:54, lambda0);
plot(1:54, lambda1);
stem(1:54, test_X(missind(6),:));
title(['Prediction probabilities (1, 0): ' num2str( probs_test(missind(3),:) ) ])
ylabel('Feature Value');
legend('\lambda_0', '\lambda_1', ['Truth label: ' num2str(test_Y(missind(3)))])
set(gca,'Xtick',1:54);
set(gca,'XtickLabel',xlabels);
xtickangle(60);
axis tight
hold off;

[sorted_amb, sorted_I] = sort( abs( probs_test(:,1) - 0.5 ) );

figure(2); clf;

subplot(3,1,1);
hold on;
plot(1:54, lambda0);
plot(1:54, lambda1);
stem(1:54, test_X(sorted_I(1),:));
title(['Prediction probabilities (1, 0): ' num2str( probs_test(sorted_I(1),:) ) ])
ylabel('Feature Value');
legend('\lambda_0', '\lambda_1', ['Truth label: ' num2str(test_Y(sorted_I(1)))])
set(gca,'Xtick',1:54);
set(gca,'XtickLabel',xlabels);
xtickangle(60);
axis tight
hold off;

subplot(3,1,2);
hold on;
plot(1:54, lambda0);
plot(1:54, lambda1);
stem(1:54, test_X(sorted_I(2),:));
title(['Prediction probabilities (1, 0): ' num2str( probs_test(sorted_I(2),:) ) ])
ylabel('Feature Value');
legend('\lambda_0', '\lambda_1', ['Truth label: ' num2str(test_Y(sorted_I(2)))])
set(gca,'Xtick',1:54);
set(gca,'XtickLabel',xlabels);
xtickangle(60);
axis tight
hold off;

subplot(3,1,3);
hold on;
plot(1:54, lambda0);
plot(1:54, lambda1);
stem(1:54, test_X(sorted_I(6),:));
title(['Prediction probabilities (1, 0): ' num2str( probs_test(sorted_I(3),:) ) ])
ylabel('Feature Value');
legend('\lambda_0', '\lambda_1', ['Truth label: ' num2str(test_Y(sorted_I(3)))])
set(gca,'Xtick',1:54);
set(gca,'XtickLabel',xlabels);
xtickangle(60);
axis tight
hold off;
