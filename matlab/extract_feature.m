function [F_cover, F_stego, names] = extract_feature(cover_dir,stego_dir)
    imgs = dir(cover_dir);
    len = length(imgs);
    F_cover = single(zeros(len-2,34671)); % SRM
    % F_cover = single(zeros(len-2,686)); % SPAM
    % F_cover = single(zeros(len-2,17000)); % GFR
    F_stego = F_cover;
    map = ones(256,256);
    names = cell(len-2,1);
    j = str2double(getenv('SLURM_CPUS_PER_TASK'));
    parpool(j);

    fprintf('\t Completion: ');
    % showTimeToCompletion; startTime = tic;
    % parfor_progress(len-2);

    parfor i = 3:len
        name = imgs(i).name;
        cover_img = [cover_dir,name];
        stego_img = [stego_dir,name];
        % F_cover(i-2,:) = GFR(cover_img,32,75);
        % F_stego(i-2,:) = GFR(stego_img,32,75);
        % F_cover(i-2,:) = featureMerge(SRM({cover_img}));
        % try
        %     F_stego(i-2,:) = featureMerge(SRM({stego_img}));
        % catch
        % cover = imread(cover_img);
        % F_cover(i-2,:) = featureMerge(maxSRM(cover, map));
        try
            stego = imread(stego_img);
            F_stego(i-2,:) = featureMerge(maxSRM(stego, map));
        catch
            fprintf('ERROR\n');
            stego_img
        end
        % end

        % try
        %     F_stego(i-2,:) = featureMerge(SRM({stego_img}));
        % catch
        %     % fprintf('ERROR\n');
        %     stego = imread(stego_img);
        %     imwrite(stego, stego_img);
        %     F_stego(i-2,:) = featureMerge(SRM({stego_img}));
        % end
        % F_cover(i-2,:) = spam686(cover_img);
        % F_stego(i-2,:) = spam686(stego_img);
        names(i-2,1) = {name}; 

        % p = parfor_progress;
        % showTimeToCompletion(p/100, [], [], startTime)
    end
    % parfor_progress(0);
    delete(gcp('nocreate'));
end