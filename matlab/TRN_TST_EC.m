function TRN_TST_EC(cover_dir, stego_dir, feature_dir, payload, ec_path, ref_trn_dir, ref_val_dir, ref_tst_dir)
    fprintf([stego_dir, '\n']);
    fprintf([feature_dir, '\n']);
    if ~exist(feature_dir,'dir'); mkdir(feature_dir); end

    %% extract features
    fprintf('feature extraction started.\n');
    % feature_extraction(cover_dir, stego_dir, feature_dir, payload);
    fprintf('feature extraction complete.\n');
    
    %% load features
    cover_fea_dir = [feature_dir, 'cover.mat'];
    split_stego = split(stego_dir, '/'); 
    % Conventional stego
    stego_fea_dir = [feature_dir, split_stego{end-2}, '_', num2str(payload), '.mat'];
    % Adversarial stego
    % stego_fea_dir = [feature_dir, split_stego{end-4}, '_', split_stego{end-3}, '_', split_stego{end-2}, '_', split_stego{end-1}, '.mat'];

    cover = matfile(cover_fea_dir);
    stego = matfile(stego_fea_dir); 
    names = cover.names;
    names = sort(names);
    cover = cover.F;
    stego = stego.F;

    % split training and testing sets (with specific split seed)
    trn_items = dir(ref_trn_dir); trn_items = trn_items(3:end);
    val_items = dir(ref_val_dir); val_items = val_items(3:end);
    trn = cell(length(trn_items)+length(val_items),1);
    for i=1:length(trn_items)
        trn{i,1} = trn_items(i).name;
    end
    for i=1:length(val_items)
        trn{i+length(trn_items),1} = val_items(i).name;
    end
    tst_items = dir(ref_tst_dir); tst_items = tst_items(3:end);
    tst = cell(length(tst_items),1);
    for i=1:length(tst_items)
        tst{i,1} = tst_items(i).name;
    end
    full_ind = 1:20000;
    trn_ind = full_ind(ismember(names, trn));
    tst_ind = full_ind(ismember(names, tst));
    TRN_cover = cover(trn_ind,:); TST_cover = cover(tst_ind,:);
    TRN_stego = stego(trn_ind,:); TST_stego = stego(tst_ind,:);

    % training
    fprintf([cover_fea_dir, '\n']);
    fprintf([stego_fea_dir, '\n']);
    [trained_ensemble,~] = ensemble_training(TRN_cover,TRN_stego);
    split_ec_path = split(ec_path, '/'); ec_dir = [];
    for i=1:length(split_ec_path)-1
        dir_cell = split_ec_path(i);
        ec_dir = [ec_dir, dir_cell{1}, '/'];
    end
    if ~exist(ec_dir,'dir'); mkdir(ec_dir); end
    save(ec_path, 'trained_ensemble');
    % testing
    test_results_cover = ensemble_testing(TST_cover,trained_ensemble);
    test_results_stego = ensemble_testing(TST_stego,trained_ensemble);
    false_alarms = sum(test_results_cover.predictions~=-1);
    missed_detections = sum(test_results_stego.predictions~=+1);

    % print
    fprintf('False Alarm Rate: %f\n', false_alarms/length(tst_ind));
    fprintf('Missed Detection Rate: %f\n', missed_detections/length(tst_ind));
    fprintf('Average Detection Error Rate: %f\n', (false_alarms+missed_detections)/(2*length(tst_ind)))
end