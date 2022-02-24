function ModClassifier = initModBoost(N)
ModClassifier.nWC = 0;
ModClassifier.WeakClas = cell(N,1);
ModClassifier.Weight = zeros(N,1);
ModClassifier.trnErr = zeros(N, 1);
ModClassifier.tstErr = zeros(N, 1);
ModClassifier.hasTestData = false;
end