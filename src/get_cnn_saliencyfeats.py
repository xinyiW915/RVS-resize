# Author: Xinyi Wang
# Date: 2021/10/05

import pandas as pd
import numpy as np
import scipy.stats
import scipy.io
import os

datasource = 'YOUTUBE_UGC'
resolution = '1080P_test'
feature_name = f'CNN_features_{resolution}'
algorithm = 'VSFACNN_Saliency'

cnnfeats = f'../CNN_features/CNN_feat_csv/{feature_name}_saliency_resize_sam.csv'
feat = pd.read_csv(cnnfeats)

feat = feat.drop(columns=['vid']) #YOUTUBE_UGC
# feat = feat.drop(columns=['flickr_id']) #KONVID_1K
print(len(feat))
print(feat)

ugc_feats = []

for i in range(len(feat)):
    ugc_feats.append(feat.loc[i])

# print(len(ugc_feats))
scipy.io.savemat(f'../feat_file/{datasource}_{resolution}_{algorithm}_feats.mat', mdict={'feats_mat': ugc_feats})