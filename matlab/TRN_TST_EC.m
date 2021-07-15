function TRN_TST_EC(cover_dir, stego_dir, feature_dir, payload, ec_path)
    fprintf([stego_dir, '\n']);
    fprintf([feature_dir, '\n']);
    if ~exist(feature_dir,'dir'); mkdir(feature_dir); end

    %% extract features
    fprintf('feature extraction started.\n');
    feature_extraction(cover_dir, stego_dir, feature_dir, payload);
    fprintf('feature extraction complete.\n');
    
    %% load features
    % common cover
    cover_fea_dir = [feature_dir, 'cover.mat'];
    % % pseudo cover
    % split_cover = split(cover_dir, '/');
    % cover_fea_dir = [feature_dir, split_cover{end-2}, '_', split_cover{end-1}, '.mat'];

    split_stego = split(stego_dir, '/'); 
    %%%% Conventional stego
    % stego_fea_dir = [feature_dir, split_stego{end-2}, '_', split_stego{end-1}, '.mat'];
    % stego_fea_dir = [feature_dir, split_stego{end-2}, '_', num2str(payload), '.mat'];
    %%%% Adversarial stego
    % stego_fea_dir = [feature_dir, split_stego{end-4}, '_', split_stego{end-3}, '_', split_stego{end-2}, '_', split_stego{end-1}, '.mat'];
    % stego_fea_dir = [feature_dir, split_stego{end-4}, '_', split_stego{end}, '.mat'];
    % stego_fea_dir = [feature_dir, split_stego{end-5}, '_', split_stego{end-3}, '_', split_stego{end-1}, '.mat'];
    %%%% min max
    % stego_fea_dir = [feature_dir, split_stego{end-6}, '_', split_stego{end-4}, '_', split_stego{end-2}, '_', split_stego{end-1}, '.mat'];
    stego_fea_dir = [feature_dir, split_stego{end-4}, '_', split_stego{end-3}, '_', split_stego{end-2}, '_', split_stego{end-1}, '.mat'];
    %%%% ADV-EMB Beta Ctrl
    % stego_fea_dir = [feature_dir, split_stego{end-4}, '_', split_stego{end-1}, '.mat'];

    cover = matfile(cover_fea_dir);
    stego = matfile(stego_fea_dir); 
    names = cover.names;
    names = sort(names);
    cover = cover.F;
    stego = stego.F;

    % RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
    % random_permutation = randperm(size(cover,1));
    % training_set = random_permutation(1:round(size(cover,1)/2));
    % testing_set = random_permutation(round(size(cover,1)/2)+1:end);

    % TRN_cover = cover(training_set,:); TST_cover = cover(testing_set,:);
    % TRN_stego = stego(training_set,:); TST_stego = stego(testing_set,:);

    % split training and testing sets (with specific split seed)
    training_names = matfile('training_names.mat'); trn_names = training_names.training_names;
    testing_names = matfile('testing_names.mat'); tst_names = testing_names.testing_names;
    full_ind = 1:20000;
    trn = cell(length(trn_names),1);
    for i=1:length(trn_names)
        name = trn_names(i,:);
        % % jpg
        % pre_name = split(name,'.'); pre_name = pre_name{1};
        % name = [pre_name, '.jpg'];

        trn(i,1) = {name(~isspace(name))};
    end
    trn_ind = full_ind(ismember(names, trn));
    tst = cell(length(tst_names),1);
    for i=1:length(tst_names)
        name = tst_names(i,:);
        % % jpg
        % pre_name = split(name,'.'); pre_name = pre_name{1};
        % name = [pre_name, '.jpg'];

        tst(i,1) = {name(~isspace(name))};
    end
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
    % fprintf('False Alarm Rate: %f\n', false_alarms/length(tst_ind));
    % fprintf('Missed Detection Rate: %f\n', missed_detections/length(tst_ind));
    fprintf('False Alarm Rate: %f\n', false_alarms/length(tst_ind));
    fprintf('Missed Detection Rate: %f\n', missed_detections/length(tst_ind));
    fprintf('Average Detection Error Rate: %f\n', (false_alarms+missed_detections)/(2*length(tst_ind)))


    % 10 different random split

    % cover_names = cover.names(ismember(cover.names,names));
    % [~,ix] = sort(cover_names);
    % C = cover.F(ismember(cover.names,names),:);
    % C = C(ix,:);
    % %% stego features
    % stego_names = stego.names(ismember(stego.names,names));
    % [~,ix] = sort(stego_names);
    % S = stego.F(ismember(stego.names,names),:);
    % S = S(ix,:);
    % %% training and testing (10 times shuffle)
    % testing_errors = zeros(1,10);
    % false_alarms_rate = zeros(1,10);
    % missed_detections_rate = zeros(1,10);
    % fid=fopen([pwd,'/','result.txt'], 'at');
    % fprintf(fid,'%s\n','**************************');
    % fprintf(fid,'%s\n',datestr(now,31));
    % fprintf(fid,'%s\n',num2str(payload));
    % for seed = 1:10
    %     %% splitting training and testing features
    %     RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed));
    %     random_permutation = randperm(size(C,1));
    %     training_set = random_permutation(1:round(size(C,1)/2));
    %     testing_set = random_permutation(round(size(C,1)/2)+1:end);
    %     TRN_cover = C(training_set,:);
    %     TRN_stego = S(training_set,:);
    %     TST_cover = C(testing_set,:);
    %     TST_stego = S(testing_set,:);
    %     %% training EC
    %     [trained_ensemble,~] = ensemble_training(TRN_cover,TRN_stego);
    %     %% testing
    %     test_results_cover = ensemble_testing(TST_cover,trained_ensemble);
    %     test_results_stego = ensemble_testing(TST_stego,trained_ensemble);
    %     %% get results (false alarm, missed detection and average testing error)
    %     false_alarms = sum(test_results_cover.predictions~=-1);
    %     missed_detections = sum(test_results_stego.predictions~=+1);
    %     %% summary
    %     num_testing_samples = size(TST_cover,1)+size(TST_stego,1);
    %     testing_errors(seed) = (false_alarms + missed_detections)/num_testing_samples;
    %     false_alarms_rate(seed) = false_alarms/num_testing_samples*2;
    %     missed_detections_rate(seed) = missed_detections/num_testing_samples*2;
    %     %% print result
    %     fprintf('Testing error %i: %.4f\n',seed,testing_errors(seed));
    % end
    % %% print all the results
    % fprintf('%s\n',stego_dir);
    % fprintf('-----------------------------\nAverage testing error over 10 splits: %.4f (+/- %.4f)\n',mean(testing_errors),std(testing_errors));
    % fprintf('---\nAverage false_alarms_rate over 10 splits: %.4f (+/- %.4f)\n',mean(false_alarms_rate),std(false_alarms_rate));
    % fprintf('---\nAverage missed_detections_rate over 10 splits: %.4f (+/- %.4f)\n',mean(missed_detections_rate),std(missed_detections_rate));
end