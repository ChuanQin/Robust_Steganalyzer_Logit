function feature_extraction(cover_dir, stego_dir, feature_dir, payload)
%% extract and save cover and stego feature
fprintf('extract cover and stego feature...\n');
[F1, F2, names] = extract_feature(cover_dir, stego_dir);
fprintf('cover and stego feature extracted.\n');
% F = F1;
% save([feature_dir,'/cover'],'F','names','-v7.3');

% split_cover = split(cover_dir, '/');
% save([feature_dir, split_cover{end-2}, '_payload_', num2str(payload), '.mat'],'F','names','-v7.3');
clear F;
F = F2;
split_stego = split(stego_dir, '/');
% stego_fea_dir = [feature_dir, split_stego{end-1}, '_', num2str(payload), '.mat', '-v7.3'];

% conventional stego
% save([feature_dir, split_stego{end-2}, '_', num2str(payload), '.mat'],'F','names','-v7.3');

% % adversarial stego
% save([feature_dir, split_stego{end-4}, '_', split_stego{end-3}, '_', split_stego{end-2}, '_', split_stego{end-1}, '.mat'],'F','names','-v7.3');
% save([feature_dir, split_stego{end-4}, '_', split_stego{end-3}, '_', num2str(payload), '.mat'],'F','names','-v7.3');
% save([feature_dir, split_stego{end-5}, '_', split_stego{end-3}, '_payload_', num2str(payload), '.mat'],'F','names','-v7.3');
% save([feature_dir, split_stego{end-4}, '_', split_stego{end-1}, '.mat'],'F','names','-v7.3');
save([feature_dir, split_stego{end-4}, '_', split_stego{end-3}, '_', split_stego{end-2}, '_', split_stego{end-1}, '.mat'],'F','names','-v7.3');
% save([feature_dir, split_stego{end-4}, '_', split_stego{end-3}, '_', split_stego{end-1}, '.mat'],'F','names','-v7.3');
% save([feature_dir, split_stego{end-5}, '_', split_stego{end-4}, '.mat'],'F','names','-v7.3');

%%%% min max
% save([feature_dir, split_stego{end-6}, '_', split_stego{end-4}, '_', split_stego{end-2}, '_', split_stego{end-1}, '.mat'],'F','names','-v7.3');

% QAS Threshold Optimization
% save([feature_dir, split_stego{end-4}, '_', split_stego{end-1}, '.mat'],'F','names','-v7.3');
clear F;
end