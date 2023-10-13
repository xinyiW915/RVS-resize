%%
% Compute features for a set of video files from datasets
%
close all;
clear;

% add path
addpath(genpath('../include/'));

% pyenv('Version', '/Users/xxxyy/opt/anaconda3/python.app/Contents/MacOS/python'); %mac
% pyenv('Version', 'C:\Users\um20242\Anaconda3\python.exe') %win

mod = py.importlib.import_module('get_cnnsafeats'); % 加载 python 模块 untitled.py，这是你自己写的模块文件
% ffmpegPath = '/usr/local/Cellar/ffmpeg/4.4_2/bin/'; %mac
%ffmpegPath = "C:\Users\um20242\ffmpeg\bin\"; %win
%addpath( ffmpegPath)

%%
% parameters
algo_name = 'NSS_VSFACNN'; % algorithm name, eg, 'V-BLIINDS'
data_name = 'YOUTUBE_UGC_1080P_test';  % dataset name, eg, 'KONVID_1K'
write_file = true;  % if true, save features on-the-fly
log_level = 0;  % 1=verbose, 0=quite

if strcmp(data_name, 'YOUTUBE_UGC')
    root_path = '/user/work/um20242/dataset/ugc-dataset/';
    data_path = '/user/work/um20242/dataset/ugc-dataset/';
elseif strcmp(data_name, 'YOUTUBE_UGC_360P')
     root_path = '/user/work/um20242/dataset/ugc-dataset/';
     data_path = '/user/work/um20242/dataset/ugc-dataset/360P';
%    root_path = "D:\ugc-dataset\ugc\original_videos\";
%    data_path = "D:\ugc-dataset\ugc\original_videos\360P\";
elseif strcmp(data_name, 'YOUTUBE_UGC_480P')
     root_path = '/user/work/um20242/dataset/ugc-dataset/';
     data_path = '/user/work/um20242/dataset/ugc-dataset/480P';
%    root_path = "D:\ugc-dataset\ugc\original_videos\";
%    data_path = "D:\ugc-dataset\ugc\original_videos\480P\";
elseif strcmp(data_name, 'YOUTUBE_UGC_720P')
     root_path = '/user/work/um20242/dataset/ugc-dataset/';
     data_path = '/user/work/um20242/dataset/ugc-dataset/720P';
%    root_path = "D:\ugc-dataset\ugc\original_videos\";
%    data_path = "D:\ugc-dataset\ugc\original_videos\720P\";
elseif strcmp(data_name, 'YOUTUBE_UGC_1080P')
     root_path = '/user/work/um20242/dataset/ugc-dataset/';
     data_path = '/user/work/um20242/dataset/ugc-dataset/2160P';
%    root_path = "D:\ugc-dataset\ugc\original_videos\";
%    data_path = "D:\ugc-dataset\ugc\original_videos\1080P\";
elseif strcmp(data_name, 'YOUTUBE_UGC_2160P')
     root_path = '/user/work/um20242/dataset/ugc-dataset/';
     data_path = '/user/work/um20242/dataset/ugc-dataset/2160P';
%    root_path = "D:\ugc-dataset\ugc\original_videos\";
%    data_path = "D:\ugc-dataset\ugc\original_videos\2160P\";
elseif strcmp(data_name, 'KONVID_1K')
     root_path = '/user/work/um20242/dataset/KoNViD_1k/';
     data_path = '/user/work/um20242/dataset/KoNViD_1k/KoNViD_1k_videos';
%    root_path = "C:\Users\um20242\OneDrive - University of Bristol\Documents\PycharmProjects\UoB\dataset\KoNViD_1k\";
%    data_path = "C:\Users\um20242\OneDrive - University of Bristol\Documents\PycharmProjects\UoB\dataset\KoNViD_1k\KoNViD_1k_videos\";
elseif strcmp(data_name, 'YOUTUBE_UGC_1080P_test')
     root_path = '/user/work/um20242/dataset/ugc-dataset/';
     data_path = '/user/work/um20242/dataset/ugc-dataset/1080P';
end

%%
% create temp dir to store decoded videos
video_tmp = '../../tmp/temp/';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../../mos_file/';
filelist_csv = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist,1);
out_path = '../../feat_file/';
if ~exist(out_path, 'dir'), mkdir(out_path); end
out_mat_name = fullfile(out_path, [data_name,'_',algo_name,'_feats.mat']);
feats_mat = [];
feats_mat_frames = cell(num_videos, 1);
%===================================================

% init deep learning models
minside = 512.0;
% net = resnet50;
% layer = 'avg_pool';

%% extract features
% parfor i = 1:num_videos % for parallel speedup
time_table = table('Size',[0, 3], 'VariableTypes', {'double', 'cell', 'double'}, 'VariableNames', {'Sequence', 'VideoName', 'RunTime'});
for i = 1:num_videos
    progressbar(i/num_videos) % Update figure
    if strcmp(data_name, 'YOUTUBE_UGC')
        video_name = fullfile(data_path, ...
            [num2str(filelist.resolution(i)),'P'],[filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
     elseif strcmp(data_name, 'YOUTUBE_UGC_360P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_480P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_720P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_1080P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_2160P')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_ALL')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'KONVID_1K')
        newStr = erase(filelist.flickr_id{i},'.mp4');
        video_name = fullfile(data_path, [newStr, '.mp4']);
        yuv_name = fullfile(video_tmp, [newStr, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC_1080P_test')
        video_name = fullfile(data_path, [filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    end
    fprintf('\n\nComputing features for %d sequence: %s\n', i, video_name);

    % decode video and store in temp dir
     cmd1 = ['ffmpeg -loglevel error -y -i ', video_name, ...
         ' -pix_fmt yuv420p -vsync 0 ', yuv_name]; %mac
     system(cmd1);
%win
%    video_name = strcat('"', video_name, '"');
%    cmd = strcat(ffmpegPath, 'ffmpeg.exe -loglevel error -y -i ');
%    cmd = strcat(cmd, {32}, video_name, ' -pix_fmt yuv420p -vsync 0 ');
%    cmd = strcat(cmd, {32}, yuv_name); %win
%    system(cmd);

    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));

    % get cnn saliency features
    tStart = tic;
    mat_file = ['../../feat_file/', data_name,'_','VSFACNN','_feats.mat'];
    cnn_num = i-1;
    cnnsa_feats = py.get_cnnsafeats.getCsf(mat_file, cnn_num);
    sl = load('../../tmp/tempmat_path/cnnsa_feats.mat');
    names = fieldnames(sl); % 获取mat中所有变量的名字
    cnnsa = sl.(names{1});

    % calculate video features
    feats_frames = calc_RAPIQUE_NSS_VSFACNN(yuv_name, width, height, ...
        framerate, minside, log_level);
    elapsed_time = toc(tStart);
    fprintf('\nOverall %f seconds elapsed...\n', elapsed_time);

    new_row = table(i, {video_name}, elapsed_time, 'VariableNames', {'Sequence', 'VideoName', 'RunTime'});
    time_table = [time_table; new_row];

    %
    feats_mat(i,:) = [nanmean(feats_frames) cnnsa];
    feats_mat_frames{i} = feats_frames;
    % clear cache
    delete(yuv_name)

    if write_file
        save(out_mat_name, 'feats_mat');
%         save(out_mat_name, 'feats_mat', 'feats_mat_frames');
        time_table_csv = fullfile(out_path, [data_name,'_',algo_name,'_runtimes.csv']);
        writetable(time_table, time_table_csv);
    end
    delete('../../tmp/tempmat_path/cnnsa_feats.mat');
end




