 MinLeafSize = 1;
        MinParentSize = 2;
        NumVariablesToSample = 'all';
        ScoreTransform = 'none';
        PruneCriterion= 'error' ;
        SplitCriterion = 'gdi';
        N = size(X, 1); % Number of training samples
        sampleWeight = repmat(1/N, N, 1);
        Weights = sampleWeight; % Train the weak learner by weights Pt
        tree = fitctree(X,y, 'MinLeafSize',MinLeafSize, 'MinParentSize', MinParentSize, 'NumVariablesToSample', NumVariablesToSample, ...
                             'PruneCriterion',PruneCriterion, 'SplitCriterion', SplitCriterion, 'Weights', Weights, 'ScoreTransform', ScoreTransform);
        prune_tree = prune(tree, 'Level', max(tree.PruneList)-1); % prune tree to have decision stump
        % if the prune tree still has more than one decision node (three inner nodes) use the max(tree.PruneList) to reduce it to just one node
        if length(prune_tree.NodeSize) > 3
            prune_tree = prune(tree, 'Level', max(tree.PruneList));
        end