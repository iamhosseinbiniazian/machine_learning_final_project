clc;
clear all;
T = readtable('bank-full.csv');
%%%%%%%%%%%%%%%%%Shuffling%%%%%%%%%%%%%%%%%
m=randperm(size(T,1),size(T,1));
T=T(m,:);
m=randperm(size(T,1),size(T,1));
T=T(m,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=T(:,1:16);
Y=T(:,17);
y=[];
for i=1:size(Y,1)
    if strcmp(table2cell(Y(i,1)),'no')==1
        y(i,1)=-1;
    else
      y(i,1)=1;  
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%Start
% Mdl = fitcnb(X,y,'ClassNames',[-1,1]);
% isLabels1 = resubPredict(Mdl);
% ConfusionMat1 = confusionmat(y,isLabels1);
disp('Start');
nfold = 5;
iter = 4;
tstError = zeros(nfold, iter);
trnError = zeros(nfold, iter);
[trnM, tstM] = buildCVMatrix(size(X, 1), nfold);
ConMat=zeros(2,2);
for n = 1:nfold
    fprintf('\tFold %d\n', n);
    idx_trn = logical(trnM(:, n) == 1);
    trnX = X(idx_trn, :);
    tstX = X(~idx_trn, :);
    trnY = y(idx_trn);
    tstY = y(~idx_trn);
    ModClassifier = buildModBoost(trnX, trnY, iter, tstX, tstY);
     [Label, Err] = predModBoost(ModClassifier, tstX, tstY);
        ConfusionMat1 = confusionmat(tstY,Label);
        ConMat=ConMat+ConfusionMat1;
    trnError(n, :) = ModClassifier.trnErr;
    tstError(n, :) = ModClassifier.tstErr;
end

% plot(1:iter, mean(trnError, 1));
% hold on;
% plot(1:iter, mean(tstError, 1));
% legend('Train','Test');