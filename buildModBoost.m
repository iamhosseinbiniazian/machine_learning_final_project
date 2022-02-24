function ModClassifier = buildModBoost(trainX, trainY, iteration, testX, testY)
if nargin < 4
    testX = [];
    testY = [];
end
ModClassifier = initModBoost(iteration);

N = size(trainX, 1); % Number of training samples
sampleWeight = repmat(1/N, N, 1);
trainX1=trainX;
trainY1=trainY;
i=1;
while i<=iteration
    
    %weakClassifier = buildStump(trainX, trainY, sampleWeight);
    disp(['Itration=',num2str(i)]);
    R = randsample([1:1:N],N,true,sampleWeight);
    trainX=trainX1(R,:);
    trainY=trainY1(R,:);
   weakClassifier=fitcnb(trainX,trainY,'ClassNames',[-1,1]);
    
    % Compute the weight of this classifier
%     isLabels1 = resubPredict(weakClassifier);
    isLabels1=predict(weakClassifier,trainX1);
    err_label = logical(trainY1 ~= isLabels1);
    err_n = sum(err_label.*sampleWeight(R))/sum(sampleWeight(R));
    if err_n<0.5
    ModClassifier.WeakClas{i} = weakClassifier;
    ModClassifier.nWC = i;
    ModClassifier.Weight(i) = 0.5*log((1-err_n)/err_n);
    % Update sample weight
%     label = predStump(trainX, weakClassifier);
%     tmpSampleWeight = -1*ModClassifier.Weight(i)*(trainY.*label); % N x 1
  tmpSampleWeight=sampleWeight;
        for j=1:N
            tmpSampleWeight(R(j)) = sampleWeight(R(j)).*exp(err_n); % N x 1
        end
    %tmpSampleWeight = sampleWeight.*exp(err_n); % N x 1
    sampleWeight = tmpSampleWeight./sum(tmpSampleWeight); % Normalized
    
    % Predict on training data
    [ttt, ModClassifier.trnErr(i)] = predModBoost(ModClassifier, trainX1, trainY1);
    % Predict on test data
    if ~isempty(testY)
        ModClassifier.hasTestData = true;
        [ttt, ModClassifier.tstErr(i)] = predModBoost(ModClassifier, testX, testY);
    end
    i=i+1;
    end
    % fprintf('\tIteration %d, Training error %f\n', i, abClassifier.trnErr(i));
end
end