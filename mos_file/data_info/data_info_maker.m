% https://www.mathworks.com/help/matlab/ref/save.html
% save your mat file with v7.3
% To view or set the default version for MAT-files, go to the Home tab and in the Environment section, click  Preferences. 
% Select MATLAB > General > MAT-Files and then choose a MAT-file save format option.

clear,clc;

%% YOUTUBE_UGC
data_path = '../../mos_file/YOUTUBE_UGC_1080P_test_metadata.csv';

%for konvid
%data = readtable(data_path);
%video_names = data.flickr_id; % video names
%scores = data.mos; % subjective scores
%height = data.height;
%width = data.width;
%framerate = data.framerate;
%clear data_path data

%for youtubeUGC
data = readtable(data_path);
video_names = data.vid; % video names
scores = data.MOSFull; % subjective scores
height = data.height;
width = data.width;
framerate = data.framerate;
clear data_path data

max_len = 1200; % maximum video length in the dataset
video_format = 'RGB'; % video format
ref_ids = [1:length(scores)]'; % video content ids
% `random` train-val-test split index, 1000 runs
index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
    1:1000,'UniformOutput', false)');
save('YOUTUBE_UGC_1080P_test_info','-v7.3')