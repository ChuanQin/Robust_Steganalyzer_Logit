function get_hand_prob_RS(feature_path, clf_path, eval_names)
    if nargin < 3
        eval_names = './testing_names.mat'
    end
    %% load features
    feature = matfile(feature_path);
    F = feature.F; 
    full_names = feature.names; 
    eval_names = matfile(eval_names); eval_names = eval_names.testing_names;
    names = full_names(ismember(full_names, eval_names));
    F = F(ismember(full_names, eval_names),:);

    %% extract features
    % [F, names] = extract_feature_demo(file_dir);
    % save('./demo_test/features/SRM.mat','F','names','-v7.3');

    %% load classifier
    ensemble_classifier = matfile(clf_path); ensemble_classifier = ensemble_classifier.trained_ensemble;

    %% get prob outputs
    results = ensemble_testing(F, ensemble_classifier);
    prob = results.probability;
    % post processing probabilities
    prob = exp(prob(:,1)*5.5)./(exp(prob(:,1)*5.5)+exp(prob(:,2)*5.5));
    % prob = prob(:,1);
    % save prob results
    save('./tmp/hand_prob.mat', 'prob', 'names');
end