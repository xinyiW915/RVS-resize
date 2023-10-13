import numpy as np
import os
import scipy.io
import pandas
from sklearn.model_selection import train_test_split
npy_feats_train = np.load('/mnt/storage/home/um20242/scratch/RVS-resize/train_test/train_COMBINED_RAPIQUE_feats.npy', allow_pickle=True)
npy_scores_train = np.load('/mnt/storage/home/um20242/scratch/RVS-resize/train_test/train_COMBINED_RAPIQUE_scores.npy', allow_pickle=True)
npy_feats_test = np.load('/mnt/storage/home/um20242/scratch/RVS-resize/train_test/test_COMBINED_RAPIQUE_feats.npy', allow_pickle=True)
npy_scores_test = np.load('/mnt/storage/home/um20242/scratch/RVS-resize/train_test/test_COMBINED_RAPIQUE_scores.npy', allow_pickle=True)
print(npy_feats_test)
print(len(npy_feats_test))
print(npy_feats_train)
print(len(npy_feats_train))
print(npy_scores_test)
print(len(npy_scores_test))
print(npy_scores_train)
print(len(npy_scores_train))
# train_data = 'train_YOUTUBE_UGC'  # 'train_KONVID_1K/train_YOUTUBE_UGC/train_COMBINED'
# test_data = 'test_YOUTUBE_UGC'  # 'test_KONVID_1K/test_YOUTUBE_UGC/test_COMBINED'
# algo_name = 'RAPIQUE'
# resolution_name = 'YOUTUBE_UGC'  # 'KONVID_1K/YOUTUBE_UGC/COMBINED'
# # read npy for train_test_split
# npy_feats_test = np.load('../train_test/' + test_data + '_' + algo_name + '_' + 'feats.npy', allow_pickle=True)
# mat_file = os.path.join('../feat_file/', resolution_name + '_' + algo_name + '_feats.mat')
# X_mat = scipy.io.loadmat(mat_file)
# X1 = np.asarray(X_mat['feats_mat'])
#
# test_data1 = 'KONVID_1K'
# csv_file1 = '../mos_file/' + test_data1 + '_metadata.save_csv'
# test_data2 = 'YOUTUBE_UGC'
# csv_file2 = '../mos_file/' + test_data2 + '_metadata.save_csv'
# d1 = pandas.read_csv(csv_file1)
# video_name = []
# for i in range(len(d1)):
#     video_name.append(d1['flickr_id'][i])
# print(len(video_name))
# d2 = pandas.read_csv(csv_file2)
# for i in range(len(d2)):
#     video_name.append(d2['vid'][i])
# print(len(video_name))
#
#
# test_feats = []
# for i in range(len(npy_feats_test)):
#     test_feats.append(npy_feats_test[i])
#
# feats = []
# for i in range(len(X1)):
#     feats.append(X1[i])
#
#
# match = {'video_name': video_name}
#
# df = pandas.DataFrame(match)
# print(df)

# for j in range(len(test_feats)):
#     for i in range(len(df)):
#         if (df['feats'][i] == test_feats[j]).all():
#             test_video.append(df['video_name'][i])
# print(len(test_video))
# print(test_video)
