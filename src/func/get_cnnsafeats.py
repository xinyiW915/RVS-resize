# Author: Xinyi Wang
# Date: 2021/10/05

import scipy.io as io
import numpy as np

def getCsf(mat_file, video_num):

    video_num = int(video_num)
    X_mat = io.loadmat(mat_file)
    X3 = np.asarray(X_mat['feats_mat'], dtype=np.float)
    # print(X3)
    # print(type(X3))
    cnnsa_feats = X3[video_num]
    io.savemat('../../tmp/tempmat_path/cnnsa_feats.mat', {'cnnsa_feats': cnnsa_feats})

    return cnnsa_feats

if __name__ == "__main__":
    mat_file = '/mnt/storage/home/um20242/scratch/RAPIQUE-VSFA-Saliency/feat_file/YOUTUBE_UGC_360P_RAPIQUE_VSFACNN_SALIENCY_feats.mat'
    cnnsa_feats = getCsf(mat_file, 1)
    print(cnnsa_feats)
