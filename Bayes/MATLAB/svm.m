clear all;

% Define fixed parameters

% Read in the files
FILE = 'ADNIMERGE_FILTERED_NOHEAD.csv';

DATA = csvread(FILE);
XDATA = DATA(:, 2:end);
YDATA = DATA(:, 1);

N = length( YDATA );
d = size( XDATA, 1 );

%% Define test/train data set splits
[trainInd, ~, testInd] = dividerand(N, 0.7, 0, 0.3);

XDATA_test = XDATA(testInd, :);
YDATA_test = YDATA(testInd);
XDATA_train = XDATA(trainInd, :);
YDATA_train = YDATA(trainInd);

SVMModel = fitcsvm(XDATA_train, YDATA_train,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

[pred_label, ~ ] = predict(SVMModel, XDATA_test);
test_accuracy = 1 - sum((pred_label - YDATA_test))/length(pred_label);

[pred_label, ~ ] = predict(SVMModel, XDATA_train);
train_accuracy = 1 - sum((pred_label - YDATA_train))/length(pred_label);

disp(['Train Accuracy: ', num2str(train_accuracy)])
disp(['Test Accuracy: ', num2str(test_accuracy)])

% svInd = SVMModel.IsSupportVector;
% h = 0.02; % Mesh grid step size
% [X1,X2] = meshgrid(min(XDATA(:,1)):h:max(XDATA(:,1)),...
%     min(XDATA(:,2)):h:max(XDATA(:,2)));
% [~,score] = predict(SVMModel,[X1(:),X2(:)]);
% scoreGrid = reshape(score(:,1),size(X1,1),size(X2,2));


% figure(1); clf;
% hold on
% plot(XDATA(YDATA == 1,1),XDATA(YDATA == 1,2),'ro')
% plot(XDATA(YDATA == 0,1),XDATA(YDATA == 0,2),'bo')
% contour(X1,X2,scoreGrid, 1)
% title('{}')
% xlabel('Feature 2')
% ylabel('Feature 1')
% legend('AD', 'CN', 'Decision Boundary')
% hold off