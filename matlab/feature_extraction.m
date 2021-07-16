function feature_extraction(cover_dir, stego_dir, feature_dir, payload)
%% extract and save cover and stego feature
fprintf('extract cover and stego feature...\n');
[F1, F2, names] = extract_feature(cover_dir, stego_dir);
fprintf('cover and stego feature extracted.\n');
clear F;
F = F2;
split_stego = split(stego_dir, '/');

% conventional stego
save([feature_dir, split_stego{end-2}, '_', num2str(payload), '.mat'],'F','names','-v7.3');

% adversarial stego
% save([feature_dir, split_stego{end-4}, '_', split_stego{end-3}, '_', split_stego{end-2}, '_', split_stego{end-1}, '.mat'],'F','names','-v7.3');
clear F;
end