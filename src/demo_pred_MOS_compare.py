# -*- coding: utf-8 -*-
"""
This script predicts a quality score in [1,5] given a VIDEVAL feature
vector by a pretrained VIDEVAL model

Input:

- feature matrix:
    eg, features/TEST_VIDEOS_VIDEVAL_feats.mat

Output:

- predicted scores:
    eg, results/TEST_VIDEOS_VIDEVAL_pred.csv

"""
# Load libraries
from sklearn import model_selection
import os
import warnings
import time
import scipy.io
from sklearn.svm import SVR
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import rc
import pandas
import seaborn as sn
# ignore all warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# Here starts the main part of the script
#
'''======================== parameters ================================'''

model_name = 'SVR'

resolution_op = '/' #'/all/'
model_path = '../model' + resolution_op
fig_path = '../fig/compare/'

data_split = True
data_name = 'KONVID_1K'
algo_name1 = 'VSFACNN'
algo_name2 = 'VSFACNN_Saliency'
fig_op = 'ALL'

def logistic_func(X1, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X1 - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def curve_bounds(x, params, sigma):
    upper_bound = logistic_func(x, params[0] + 2 * sigma[0], params[1] + 2 * sigma[1], params[2] + 2 * sigma[2], params[3] + 2 * sigma[3])
    lower_bound = logistic_func(x, params[0] - 2 * sigma[0], params[1] - 2 * sigma[1], params[2] - 2 * sigma[2], params[3] + 2 * sigma[3])
    return upper_bound, lower_bound

if data_split == True:
    train_data = 'train_' + data_name   # 'train_KONVID_1K/train_YOUTUBE_UGC_ALL/train_COMBINED'
    test_data = 'test_' + data_name   # 'test_KONVID_1K/test_YOUTUBE_UGC_ALL/test_COMBINED'
    resolution_name = 'test_' + data_name   # 'KONVID_1K/YOUTUBE_UGC_ALL/COMBINED'

    # read npy for train_test_split
    npy_feats_test1 = np.load('../train_test' + resolution_op + test_data + '_' + algo_name1 + '_' + 'feats.npy', allow_pickle=True)
    npy_scores_test1 = np.load('../train_test' + resolution_op + test_data + '_' + algo_name1 + '_' + 'scores.npy', allow_pickle=True)
    npy_feats_test2 = np.load('../train_test' + resolution_op + test_data + '_' + algo_name2 + '_' + 'feats.npy', allow_pickle=True)
    npy_scores_test2 = np.load('../train_test' + resolution_op + test_data + '_' + algo_name2 + '_' + 'scores.npy', allow_pickle=True)
    try:
        video_name1 = npy_scores_test1[:, 0]
        mos1 = npy_scores_test1[:, 1]
        video_name2 = npy_scores_test2[:, 0]
        mos2 = npy_scores_test2[:, 1]
        if (video_name1 == video_name2).all():
            print('equal')
            video_name = video_name1 + '--' + video_name2
        else:
            print('not equal')
            video_name = video_name1 + '--' + video_name2
        if (mos1 == mos2).all():
            print('equal')
            mos = mos1 = mos2
        else:
            print('not equal')
    except:
        raise Exception('Read npy file error!')
    '''======================== read files =============================== '''
    X1 = npy_feats_test1
    X1[np.isnan(X1)] = 0
    X1[np.isinf(X1)] = 0

    X2 = npy_feats_test2
    X2[np.isnan(X2)] = 0
    X2[np.isinf(X2)] = 0

    # save model
    model_file1 = os.path.join(model_path, train_data + '_' + algo_name1 + '_trained_svr.pkl')
    scaler_file1 = os.path.join(model_path, train_data + '_' + algo_name1 + '_trained_scaler.pkl')
    pars_file1 = os.path.join(model_path, train_data + '_' + algo_name1 + '_logistic_pars.mat')

    model_file2 = os.path.join(model_path, train_data + '_' + algo_name2 + '_trained_svr.pkl')
    scaler_file2 = os.path.join(model_path, train_data + '_' + algo_name2 + '_trained_scaler.pkl')
    pars_file2 = os.path.join(model_path, train_data + '_' + algo_name2 + '_logistic_pars.mat')

    print("Predict quality scores using pretrained {} with {} on split dataset {} ...".format(train_data + '_' + algo_name1 , model_name, test_data))
    print("Predict quality scores using pretrained {} with {} on split dataset {} ...".format(train_data + '_' + algo_name2, model_name, test_data))

    '''======================== read saved model =============================== '''
    model1 = joblib.load(model_file1)
    scaler1 = joblib.load(scaler_file1)
    popt1 = np.asarray(scipy.io.loadmat(pars_file1)['popt'][0], dtype=np.float)

    model2 = joblib.load(model_file2)
    scaler2 = joblib.load(scaler_file2)
    popt2 = np.asarray(scipy.io.loadmat(pars_file2)['popt'][0], dtype=np.float)

    # Algorithm 1
    X1 = scaler1.transform(X1)
    y_pred1 = model1.predict(X1)
    y1 = logistic_func(y_pred1, *popt1)
    # print('Predicted MOS in [1,5]:')
    # print(y1)

    # Algorithm 2
    X2 = scaler2.transform(X2)
    y_pred2 = model2.predict(X2)
    y2 = logistic_func(y_pred2, *popt2)
    # print('Predicted MOS in [1,5]:')
    # print(y2)

else:
    model_data = 'COMBINED'  # 'YOUTUBE_UGC_ALL/KONVID_1K/COMBINED'
    test_data = 'KONVID_1K'     # 'KONVID_1K/YOUTUBE_UGC_ALL/COMBINED'
    resolution_name = 'KONVID_1K_testOn_COMBINED'   # 'KONVID_1K/YOUTUBE_UGC_ALL/COMBINED'

    # read mat file for all dataset
    mat_file = os.path.join('../feat_file' + resolution_op, test_data + '_' + algo_name1 + '_feats.mat')
    print(mat_file)
    mat_ori = os.path.join('../feat_file' + resolution_op, test_data + '_' + algo_name2 + '_feats.mat')
    print(mat_ori)
    '''======================== read files =============================== '''
    X_mat1 = scipy.io.loadmat(mat_file)
    X1 = np.asarray(X_mat1['feats_mat'], dtype=np.float)
    X1[np.isnan(X1)] = 0
    X1[np.isinf(X1)] = 0

    X_mat2 = scipy.io.loadmat(mat_ori)
    X2 = np.asarray(X_mat2['feats_mat'], dtype=np.float)
    X2[np.isnan(X2)] = 0
    X2[np.isinf(X2)] = 0

    # save model
    model_file1 = os.path.join(model_path, model_data + '_' + algo_name1 + '_trained_svr.pkl')
    scaler_file1 = os.path.join(model_path, model_data + '_' + algo_name1 + '_trained_scaler.pkl')
    pars_file1 = os.path.join(model_path, model_data + '_' + algo_name1 + '_logistic_pars.mat')

    model_file2 = os.path.join(model_path, model_data + '_' + algo_name2 + '_trained_svr.pkl')
    scaler_file2 = os.path.join(model_path, model_data + '_' + algo_name2 + '_trained_scaler.pkl')
    pars_file2 = os.path.join(model_path, model_data + '_' + algo_name2 + '_logistic_pars.mat')

    print("Predict quality scores using pretrained {} with {} on dataset {} ...".format(model_data + '_' + algo_name1, model_name,
                                                                                        test_data))
    print("Predict quality scores using pretrained {} with {} on dataset {} ...".format(model_data + '_' + algo_name2, model_name,
                                                                                        test_data))
    '''======================== read saved model =============================== '''
    model1 = joblib.load(model_file1)
    scaler1 = joblib.load(scaler_file1)
    popt1 = np.asarray(scipy.io.loadmat(pars_file1)['popt'][0], dtype=np.float)

    model2 = joblib.load(model_file2)
    scaler2 = joblib.load(scaler_file2)
    popt2 = np.asarray(scipy.io.loadmat(pars_file2)['popt'][0], dtype=np.float)

    # algorithm 1
    X1 = scaler1.transform(X1)
    y_pred1 = model1.predict(X1)
    y1 = logistic_func(y_pred1, *popt1)
    # print('Predicted MOS in [1,5]:')
    # print(y1)

    # algorithm 2
    X2 = scaler2.transform(X2)
    y_pred2 = model2.predict(X2)
    y2 = logistic_func(y_pred2, *popt2)
    # print('Predicted MOS in [1,5]:')
    # print(y2)

    if test_data == 'KONVID_1K':
        csv_file = '../mos_file/' + test_data + '_metadata.csv'
        try:
            d = pandas.read_csv(csv_file)
            video_name = d['flickr_id']
            mos = d['mos']
        except:
            raise Exception('Read csv file error!')
    elif test_data == 'YOUTUBE_UGC_ALL':
        csv_file = '../mos_file/'+test_data+'_metadata.csv'
        try:
            d = pandas.read_csv(csv_file)
            video_name = d['vid']
            mos = d['MOSFull']
        except:
            raise Exception('Read csv file error!')


data = {'Video_name': video_name,
        'MOS1': mos1,
        'y_pred1': y_pred1,
        "Predicted Score_" + algo_name1: y1,
        'MOS2': mos2,
        'y_pred2': y_pred2,
        "Predicted Score_" + algo_name2: y2}
result = pandas.DataFrame(data)
print(result)

# result_path = '../result' + resolution_op + resolution_name + '_' + '_Predicted_Score.csv'
# result.to_csv(result_path, index=False, header=True)

mos1 = np.array(mos1, dtype=np.float64)
mos2 = np.array(mos2, dtype=np.float64)
y1 = np.array(y1, dtype=np.float64)
y2 = np.array(y2, dtype=np.float64)

if fig_op ==algo_name1:
    print('=======================' + algo_name1 + '===============================')
    # scatter plot for one test dataset
    # nonlinear logistic fitted curve / logistic regression
    try:
        beta = [np.max(mos1), np.min(mos1), np.mean(y1), 0.5]
        popt, pcov = curve_fit(logistic_func, y1, mos1, p0=beta, maxfev=100000000)
        sigma = np.sqrt(np.diag(pcov))
    except:
        raise Exception('Fitting logistic function time-out!!')
    x_values1 = np.linspace(np.min(y1), np.max(y1), len(y1))
    upper_bound, lower_bound = curve_bounds(x_values1, popt, sigma)
    plt.plot(x_values1, logistic_func(x_values1, *popt), '-', color='#c72e29', label='Fitted f(x)')
    # plt.plot(x_values1, upper_bound, 'b--', label='f(x)±2σ')
    # plt.plot(x_values1, lower_bound, 'b--', label='f(x)±2σ')
    fig1 = sn.scatterplot(x="Predicted Score_" + algo_name1, y="MOS1", data=result, markers='o', color='steelblue', label=algo_name1)
    plt.legend(loc='upper left')
    plt.ylim(1, 5)
    plt.xlim(1, 5)
    plt.title(resolution_name, fontsize=10)
    plt.xlabel('Predicted Score')
    plt.ylabel('MOS')
    reg_fig1 = fig1.get_figure()
    reg_fig1.savefig(fig_path + resolution_name + '_' + algo_name1, dpi=400)

elif fig_op == algo_name2:
    print('========================' + algo_name2 + '==============================')
    # nonlinear logistic fitted curve / logistic regression
    try:
        beta = [np.max(mos), np.min(mos2), np.mean(y2), 0.5]
        popt, pcov = curve_fit(logistic_func, y2, mos2, p0=beta, maxfev=100000000)
        sigma = np.sqrt(np.diag(pcov))
    except:
        raise Exception('Fitting logistic function time-out!!')
    x_values2 = np.linspace(np.min(y2), np.max(y2), len(y2))
    upper_bound, lower_bound = curve_bounds(x_values2, popt, sigma)
    plt.plot(x_values2, logistic_func(x_values2, *popt), '-', color='#c72e29', label='Fitted f(x)')
    # plt.plot(x_values2, upper_bound, 'b--', label='f(x)±2σ')
    # plt.plot(x_values2, lower_bound, 'b--', label='f(x)±2σ')
    fig2 = sn.scatterplot("Predicted Score_" + algo_name2, y="MOS2", data=result, markers='X', color='darkorange', label=algo_name2)
    plt.legend(loc='upper left')
    plt.ylim(1, 5)
    plt.xlim(1, 5)
    plt.title(resolution_name, fontsize=10)
    plt.xlabel('Predicted Score')
    plt.ylabel('MOS')
    reg_fig2 = fig2.get_figure()
    reg_fig2.savefig(fig_path + resolution_name + '_' + algo_name2, dpi=400)

elif fig_op == 'ALL':
    print('=========================ALL=============================')
    # scatter plot for two test dataset
    sn.scatterplot(x="Predicted Score_" + algo_name1, y="MOS1", data=result, markers='X', color='darkorange', label=algo_name1)
    # print(sorted(result["Predicted Score_" + algo_name2].tolist()))
    fig = sn.scatterplot(x="Predicted Score_" + algo_name2, y="MOS2", data=result, markers='o', color='steelblue', label=algo_name2)
    plt.legend(loc='upper left', title="Algorithm")
    plt.ylim(1, 5)
    plt.xlim(1, 5)
    plt.title(resolution_name, fontsize=10)
    plt.xlabel('Predicted Score')
    plt.ylabel('MOS')
    reg_fig = fig.get_figure()
    reg_fig.savefig(fig_path + resolution_name + '_' + algo_name1 + '_vs_' + algo_name2, dpi=400)


# # popt, pcov = curve_fit(logistic_func, mos, y2)
# # print(popt)
# # plt.plot(mos, logistic_func(mos, *popt), 'r-')
# fig = sn.lineplot(x_values1, y=logistic_func(x_values1, *popt), markers='--', color='#c72e29', label='RS_reize', label='Fitted f(x)')
# fig = sn.lineplot(x=mos, y=logistic_func(mos, *popt), markers='--', color='#c72e29')
# plt.ylim(1, 5)
# plt.xlim(1.2, 4.70)
# plt.xlabel('Predicted Score')
# plt.ylabel('MOS')
# curve_fig = fig.get_figure()
# curve_fig.savefig(fig_path + test_data + '_' + algo_name2 + '_' + 'fitted curve', dpi=400)