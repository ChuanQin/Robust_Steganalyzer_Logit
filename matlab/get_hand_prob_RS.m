function get_hand_prob_RS(feature_path, clf_path, ref_tst_dir)
    feature = matfile(feature_path);
    F = feature.F; 
    full_names = feature.names; 
    names = feature.names; 
    if nargin == 3
        tst_items = dir(ref_tst_dir); tst_items = tst_items(3:end);
        tst = cell(length(tst_items),1);
        for i=1:length(tst_items)
            tst{i,1} = tst_items(i).name;
        end
        names = full_names(ismember(full_names, tst));
        F = F(ismember(full_names, tst),:);
    end

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