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
# import joblib
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

resolution_op = '/resolution/' #'/resolution/'
model_path = '../model' + resolution_op
fig_path = '../fig' + resolution_op

data_split = True
data_name = 'YOUTUBE_UGC_1080P'
algo_name1 = 'CNN'

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
    try:
        video_name1 = npy_scores_test1[:, 0]
        mos1 = npy_scores_test1[:, 1]
    except:
        raise Exception('Read npy file error!')
    '''======================== read files =============================== '''
    X1 = npy_feats_test1
    X1[np.isnan(X1)] = 0
    X1[np.isinf(X1)] = 0

    # save model
    model_file1 = os.path.join(model_path, train_data + '_' + algo_name1 + '_trained_svr.pkl')
    scaler_file1 = os.path.join(model_path, train_data + '_' + algo_name1 + '_trained_scaler.pkl')
    pars_file1 = os.path.join(model_path, train_data + '_' + algo_name1 + '_logistic_pars.mat')

    print("Predict quality scores using pretrained {} with {} on split dataset {} ...".format(train_data + '_' + algo_name1 , model_name, test_data))

    '''======================== read saved model =============================== '''
    model1 = joblib.load(model_file1)
    scaler1 = joblib.load(scaler_file1)
    popt1 = np.asarray(scipy.io.loadmat(pars_file1)['popt'][0], dtype=np.float)

    X1 = scaler1.transform(X1)
    y_pred1 = model1.predict(X1)
    y1 = logistic_func(y_pred1, *popt1)
    # print('Predicted MOS in [1,5]:')
    # print(y1)

data = {'Video_name': video_name1,
        'MOS': mos1,
        'y_pred1': y_pred1,
        "Predicted Score_" + algo_name1: y1}
result = pandas.DataFrame(data)
print(result)

result_path = '../result' + resolution_op + resolution_name + '_' + algo_name1 + '_Predicted_Score.csv'
result.to_csv(result_path, index=False, header=True)

mos1 = np.array(mos1, dtype=np.float64)
y1 = np.array(y1, dtype=np.float64)


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
fig1 = sn.scatterplot(x="Predicted Score_" + algo_name1, y="MOS", data=result, markers='o', color='steelblue', label=algo_name1)
plt.legend(loc='upper left')
plt.ylim(1, 5)
plt.xlim(1, 5)
plt.title(resolution_name, fontsize=10)
plt.xlabel('Predicted Score')
plt.ylabel('MOS')
reg_fig1 = fig1.get_figure()
reg_fig1.savefig(fig_path + resolution_name + '_' + algo_name1, dpi=400)



