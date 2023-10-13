# -*- coding: utf-8 -*-
"""
Apply k-folds train and validate regression model to predict MOS from the features of UGC videos dataset
and plot the figs
# Modify: Xinyi Wang
# Date: 2022/12/21

"""
import warnings
import time
import os
# ignore all warnings
warnings.filterwarnings("ignore")
# Load libraries
import pandas
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import scipy.stats
import scipy.io
import sys
import os

from sklearn import model_selection
from scipy.optimize import curve_fit
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
# import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    log_path = '../log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # ===========================================================================
    # Here starts the main part of the script
    #
    '''======================== parameters ================================'''

    model_name = 'SVR'
    algo_name = 'CNN'
    data_name = 'YOUTUBE_UGC_1080P'

    save_path = '../model/'

    ## read KONVID_1K
    if data_name == 'KONVID_1K' or data_name == 'COMBINED':
        data_name1 = 'KONVID_1K'
        csv_file = '../mos_file/'+data_name1+'_metadata.csv'
        mat_file = '../feat_file/all/'+data_name1+'_'+algo_name+'_feats.mat'
        print(mat_file)
        try:
            df1 = pandas.read_csv(csv_file)
            konvid_name = df1['flickr_id']
        except:
            raise Exception('Read csv file error!')
        y1 = df1['mos']
        y1 = np.array(list(y1), dtype=np.float)
        X_mat = scipy.io.loadmat(mat_file)
        X1 = np.asarray(X_mat['feats_mat'], dtype=np.float)
        # apply scaling transform
        temp = np.divide(5.0 - y1, 4.0) # convert MOS do distortion
        temp = -0.0993 + 1.1241 * temp # apply gain and shift produced by INSLA
        temp = 5.0 - 4.0 * temp # convert distortion to MOS
        y1 = temp

    ## read YOUTUBE_UGC
    if data_name == 'KONVID_1K' or data_name == 'YOUTUBE_UGC_ALL' or data_name == 'COMBINED':
        data_name2 = 'YOUTUBE_UGC_ALL'
        mat_file = '../feat_file/all/' + data_name2 + '_' + algo_name + '_feats.mat'
    else:
        data_name2 = data_name
        mat_file = '../feat_file/resolution/' + data_name2 + '_' + algo_name + '_feats.mat'
    csv_file = '../mos_file/' + data_name2 + '_metadata.csv'
    print(mat_file)
    try:
        df3 = pandas.read_csv(csv_file)
        youtube_name = df3['vid']
    except:
        raise Exception('Read csv file error!')
    y3 = df3['MOSFull']
    y3 = np.array(list(y3), dtype=np.float)
    X_mat = scipy.io.loadmat(mat_file)
    X3 = np.asarray(X_mat['feats_mat'], dtype=np.float)

    ## read metadata
    if data_name == 'KONVID_1K':
        # read KONVID_1K
        X = np.vstack((X1))
        y = np.vstack((y1.reshape(-1, 1))).squeeze()
        mos_scores1 = []
        for i in range(len(y)):
            mos_scores1.append(y[i])
        matching = {'video_name': konvid_name,
                    'mos_scores': mos_scores1}
        match = pandas.DataFrame(matching)
    elif data_name == 'YOUTUBE_UGC_ALL' or data_name == 'YOUTUBE_UGC_360P' or data_name == 'YOUTUBE_UGC_480P' or data_name == 'YOUTUBE_UGC_720P' \
        or data_name == 'YOUTUBE_UGC_1080P' or data_name == 'YOUTUBE_UGC_2160P':
        # read YOUTUBE_UGC
        X = np.vstack((X3))
        y = np.vstack((y3.reshape(-1,1))).squeeze()
        mos_scores3 = []
        for i in range(len(y)):
            mos_scores3.append(y[i])
        matching = {'video_name': youtube_name,
                    'mos_scores': mos_scores3}
        match = pandas.DataFrame(matching)
    else:
        # read COMBINED data
        X = np.vstack((X1, X3))
        y = np.vstack((y1.reshape(-1, 1), y3.reshape(-1, 1))).squeeze()
        video_name = konvid_name.append(youtube_name)
        mos_scores = []
        for i in range(len(y)):
            mos_scores.append(y[i])
        matching = {'video_name': video_name,
                    'mos_scores': mos_scores}
        match = pandas.DataFrame(matching)

    # #############################################################################
    # Train classifiers
    # define 4-parameter logistic regression
    def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
        logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
        yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
        return yhat

    model_params_all_repeats = []
    PLCC_all_repeats_test = []
    SRCC_all_repeats_test = []
    KRCC_all_repeats_test = []
    RMSE_all_repeats_test = []
    PLCC_all_repeats_train = []
    SRCC_all_repeats_train = []
    KRCC_all_repeats_train = []
    RMSE_all_repeats_train = []

    # For an initial search, a logarithmic grid with basis
    # 10 is often helpful. Using a basis of 2, a finer
    # tuning can be achieved but at a much higher cost.


    log_file_name = log_path + 'train_' + data_name + '_' + algo_name + '_GridSearch.log'
    # for print
    sys.stdout = Logger(log_file_name)
    # for traceback
    sys.stderr = Logger(log_file_name)

    print("Evaluating algorithm {} with {} on split dataset {} ...".format(algo_name, model_name, data_name))

    ## best parameter search
    C_range = np.logspace(1, 10, 10, base=2)
    gamma_range = np.logspace(-8, 1, 10, base=2)
    params_grid = dict(gamma=gamma_range, C=C_range)

    # 102 random splits
    for i in range(1, 102):
        print(i, 'th repeated 80-20 hold out test')
        t0 = time.time()
        # parameters for each hold out
        model_params_all = []
        PLCC_all_train = []
        SRCC_all_train = []
        KRCC_all_train = []
        RMSE_all_train = []
        PLCC_all_test = []
        SRCC_all_test = []
        KRCC_all_test = []
        RMSE_all_test = []

        # train test split
        test_size = 0.2
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, match, test_size=test_size, random_state=math.ceil(8.8 * i))
        np.save('../train_test/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_feats.npy', X_train)
        np.save('../train_test/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_scores.npy', y_train)
        np.save('../train_test/' + 'test_' + data_name + '_' + algo_name + '_' + str(i) + '_feats.npy', X_test)
        np.save('../train_test/' + 'test_' + data_name + '_' + algo_name + '_' + str(i) + '_scores.npy', y_test)
        y_train = y_train['mos_scores'].values.tolist()
        y_test = y_test['mos_scores'].values.tolist()

        Iter = 0
        # SVR grid search in the TRAINING SET ONLY
        validation_size = 0.2
        X_param_train, X_param_valid, y_param_train, y_param_valid = model_selection.train_test_split(X_train, y_train, test_size=validation_size, random_state=math.ceil(6.6 * i))

        ## preprocessing
        X_param_train[np.isinf(X_param_train)] = np.nan
        imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_param_train)
        X_param_train = imp.transform(X_param_train)
        X_param_valid[np.isinf(X_param_valid)] = np.nan
        imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_param_valid)
        X_param_valid = imp.transform(X_param_valid)

        # grid search
        for C in C_range:
            for gamma in gamma_range:
                model_params_all.append((C, gamma))
                model = SVR(kernel='rbf', gamma=gamma, C=C)
                # Standard min-max normalization of features
                scaler = preprocessing.MinMaxScaler().fit(X_param_train)
                X_param_train = scaler.transform(X_param_train)

                # Fit training set to the regression model
                model.fit(X_param_train, y_param_train)

                # Apply scaling
                X_param_valid = scaler.transform(X_param_valid)

                # Predict MOS for the validation set
                y_param_valid_pred = model.predict(X_param_valid)
                y_param_train_pred = model.predict(X_param_train)

                y_param_valid = np.array(list(y_param_valid), dtype=np.float)
                y_param_train = np.array(list(y_param_train), dtype=np.float)

                try:
                    # logistic regression
                    beta = [np.max(y_param_valid), np.min(y_param_valid), np.mean(y_param_valid_pred), 0.5]
                    popt_valid, _ = curve_fit(logistic_func, y_param_valid_pred, y_param_valid, p0=beta, maxfev=100000000)
                    y_param_valid_pred_logistic = logistic_func(y_param_valid_pred, *popt_valid)
                    # logistic regression
                    beta = [np.max(y_param_train), np.min(y_param_train), np.mean(y_param_train_pred), 0.5]
                    popt_valid, _ = curve_fit(logistic_func, y_param_valid_pred, y_param_valid, p0=beta, maxfev=100000000)
                    y_param_train_pred_logistic = logistic_func(y_param_train_pred, *popt_valid)
                except:
                    raise Exception('Fitting logistic function time-out!!')

                plcc_valid_tmp = scipy.stats.pearsonr(y_param_valid, y_param_valid_pred_logistic)[0]
                rmse_valid_tmp = np.sqrt(mean_squared_error(y_param_valid, y_param_valid_pred_logistic))
                srcc_valid_tmp = scipy.stats.spearmanr(y_param_valid, y_param_valid_pred)[0]
                krcc_valid_tmp = scipy.stats.kendalltau(y_param_valid, y_param_valid_pred)[0]
                plcc_train_tmp = scipy.stats.pearsonr(y_param_train, y_param_train_pred_logistic)[0]
                rmse_train_tmp = np.sqrt(mean_squared_error(y_param_train, y_param_train_pred_logistic))
                srcc_train_tmp = scipy.stats.spearmanr(y_param_train, y_param_train_pred)[0]

                try:
                    krcc_train_tmp = scipy.stats.kendalltau(y_param_train, y_param_train_pred)[0]
                except:
                    krcc_train_tmp = scipy.stats.kendalltau(y_param_train, y_param_train_pred, method='asymptotic')[0]
                # save results
                PLCC_all_test.append(plcc_valid_tmp)
                RMSE_all_test.append(rmse_valid_tmp)
                SRCC_all_test.append(srcc_valid_tmp)
                KRCC_all_test.append(krcc_valid_tmp)
                PLCC_all_train.append(plcc_train_tmp)
                RMSE_all_train.append(rmse_train_tmp)
                SRCC_all_train.append(srcc_train_tmp)
                KRCC_all_train.append(krcc_train_tmp)

        # using the best chosen parameters to test on testing set
        param_idx = np.argmin(np.asarray(RMSE_all_test, dtype=np.float))
        C_opt, gamma_opt = model_params_all[param_idx]
        model = SVR(kernel='rbf', gamma=gamma_opt, C=C_opt)
        print("best chosen parameters: kernel='rbf', gamma={}, C={} ...".format(gamma_opt, C_opt))

        ## preprocessing
        X_train[np.isinf(X_train)] = np.nan
        imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_train)
        X_train = imp.transform(X_train)
        X_test[np.isinf(X_test)] = np.nan
        imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_test)
        X_test = imp.transform(X_test)

        # Standard min-max normalization of features
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        # Fit training set to the regression model
        model.fit(X_train, y_train)

        # Apply scaling
        X_test = scaler.transform(X_test)

        # Predict MOS for the validation set
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        y_test = np.array(list(y_test), dtype=np.float)
        y_train = np.array(list(y_train), dtype=np.float)

        try:
            # logistic regression
            beta = [np.max(y_test), np.min(y_test), np.mean(y_test_pred), 0.5]
            popt_test, _ = curve_fit(logistic_func, y_test_pred, y_test, p0=beta, maxfev=100000000)
            y_test_pred_logistic = logistic_func(y_test_pred, *popt_test)
            # logistic regression
            beta = [np.max(y_train), np.min(y_train), np.mean(y_train_pred), 0.5]
            popt_train, _ = curve_fit(logistic_func, y_train_pred, y_train, p0=beta, maxfev=100000000)
            y_train_pred_logistic = logistic_func(y_train_pred, *popt_train)
        except:
            raise Exception('Fitting logistic function time-out!!')

        # save model
        print('======================================================')
        print('model:', model)
        print('scaler:', scaler)
        print('popt:', popt_train)
        # for train_test_split
        joblib.dump(model, os.path.join(save_path, 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_trained_svr.pkl'))
        joblib.dump(scaler, os.path.join(save_path, 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_trained_scaler.pkl'))
        scipy.io.savemat(os.path.join(save_path, 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_logistic_pars.mat'), mdict={'popt': np.asarray(popt_train, dtype=np.float)})

        plcc_test_opt = scipy.stats.pearsonr(y_test, y_test_pred_logistic)[0]
        rmse_test_opt = np.sqrt(mean_squared_error(y_test, y_test_pred_logistic))
        srcc_test_opt = scipy.stats.spearmanr(y_test, y_test_pred)[0]
        krcc_test_opt = scipy.stats.kendalltau(y_test, y_test_pred)[0]

        plcc_train_opt = scipy.stats.pearsonr(y_train, y_train_pred_logistic)[0]
        rmse_train_opt = np.sqrt(mean_squared_error(y_train, y_train_pred_logistic))
        srcc_train_opt = scipy.stats.spearmanr(y_train, y_train_pred)[0]
        krcc_train_opt = scipy.stats.kendalltau(y_train, y_train_pred)[0]

        model_params_all_repeats.append((C_opt, gamma_opt))
        SRCC_all_repeats_test.append(srcc_test_opt)
        KRCC_all_repeats_test.append(krcc_test_opt)
        PLCC_all_repeats_test.append(plcc_test_opt)
        RMSE_all_repeats_test.append(rmse_test_opt)
        SRCC_all_repeats_train.append(srcc_train_opt)
        KRCC_all_repeats_train.append(krcc_train_opt)
        PLCC_all_repeats_train.append(plcc_train_opt)
        RMSE_all_repeats_train.append(rmse_train_opt)

        # print results for each iteration
        print('======================================================')
        print('Best results in CV grid search within one split')
        print('SRCC_train: ', srcc_train_opt)
        print('KRCC_train: ', krcc_train_opt)
        print('PLCC_train: ', plcc_train_opt)
        print('RMSE_train: ', rmse_train_opt)
        print('======================================================')
        print('SRCC_test: ', srcc_test_opt)
        print('KRCC_test: ', krcc_test_opt)
        print('PLCC_test: ', plcc_test_opt)
        print('RMSE_test: ', rmse_test_opt)
        print('MODEL: ', (C_opt, gamma_opt))
        print('======================================================')

        print(' -- ' + str(time.time() - t0) + ' seconds elapsed...\n\n')

    print('\n\n')
    print('======================================================')
    print('Average training results among all repeated 80-20 holdouts:')
    print('SRCC: ', np.median(SRCC_all_repeats_train), '( std:', np.std(SRCC_all_repeats_train), ')')
    print('KRCC: ', np.median(KRCC_all_repeats_train), '( std:', np.std(KRCC_all_repeats_train), ')')
    print('PLCC: ', np.median(PLCC_all_repeats_train), '( std:', np.std(PLCC_all_repeats_train), ')')
    print('RMSE: ', np.median(RMSE_all_repeats_train), '( std:', np.std(RMSE_all_repeats_train), ')')
    print('======================================================')
    print('Average testing results among all repeated 80-20 holdouts:')
    print('SRCC: ', np.median(SRCC_all_repeats_test), '( std:', np.std(SRCC_all_repeats_test), ')')
    print('KRCC: ', np.median(KRCC_all_repeats_test), '( std:', np.std(KRCC_all_repeats_test), ')')
    print('PLCC: ', np.median(PLCC_all_repeats_test), '( std:', np.std(PLCC_all_repeats_test), ')')
    print('RMSE: ', np.median(RMSE_all_repeats_test), '( std:', np.std(RMSE_all_repeats_test), ')')
    print('======================================================')
    print('\n\n')

    # find the median model
    print('======================================================')
    print('median test RMSE: ', np.median(RMSE_all_repeats_test))  # median
    sorted_ind = np.argsort(RMSE_all_repeats_test)  # Array A is sorted in ascending order, corresponding to the index array 数组a升序排序后对应的索引数组
    indMedian = sorted_ind[len(RMSE_all_repeats_test) // 2] + 1  # median index
    print('all RMSE: ', RMSE_all_repeats_test)
    print('sorted index: ', sorted_ind)
    print('median index: ', indMedian)

    C_opt_best, gamma_opt_best = model_params_all_repeats[indMedian]
    print("best chosen parameters among all repeated 80-20 holdouts: kernel='rbf', gamma={}, C={} ...".format(gamma_opt_best, C_opt_best))
    for i in range(1, 102):
        if i != indMedian:
            os.remove('../train_test/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_feats.npy')
            os.remove('../train_test/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_scores.npy')
            os.remove('../train_test/' + 'test_' + data_name + '_' + algo_name + '_' + str(i) + '_feats.npy')
            os.remove('../train_test/' + 'test_' + data_name + '_' + algo_name + '_' + str(i) + '_scores.npy')

            os.remove('../model/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_trained_svr.pkl')
            os.remove('../model/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_trained_scaler.pkl')
            os.remove('../model/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_logistic_pars.mat')
        else:
            old_name_train_feats = '../train_test/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_feats.npy'
            old_name_train_scores = '../train_test/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_scores.npy'
            old_name_test_feats = '../train_test/' + 'test_' + data_name + '_' + algo_name + '_' + str(i) + '_feats.npy'
            old_name_test_scores = '../train_test/' + 'test_' + data_name + '_' + algo_name + '_' + str(i) + '_scores.npy'

            new_name_train_feats = '../train_test/' + 'train_' + data_name + '_' + algo_name + '_feats.npy'
            new_name_train_scores = '../train_test/' + 'train_' + data_name + '_' + algo_name + '_scores.npy'
            new_name_test_feats = '../train_test/' + 'test_' + data_name + '_' + algo_name + '_feats.npy'
            new_name_test_scores = '../train_test/' + 'test_' + data_name + '_' + algo_name + '_scores.npy'

            os.rename(old_name_train_feats, new_name_train_feats)
            os.rename(old_name_train_scores, new_name_train_scores)
            os.rename(old_name_test_feats, new_name_test_feats)
            os.rename(old_name_test_scores, new_name_test_scores)

            old_name_model = '../model/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_trained_svr.pkl'
            old_name_scaler = '../model/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_trained_scaler.pkl'
            old_name_popt = '../model/' + 'train_' + data_name + '_' + algo_name + '_' + str(i) + '_logistic_pars.mat'

            new_name_model = '../model/' + 'train_' + data_name + '_' + algo_name + '_trained_svr.pkl'
            new_name_scaler = '../model/' + 'train_' + data_name + '_' + algo_name + '_trained_scaler.pkl'
            new_name_popt = '../model/' + 'train_' + data_name + '_' + algo_name + '_logistic_pars.mat'

            os.rename(old_name_model, new_name_model)
            os.rename(old_name_scaler, new_name_scaler)
            os.rename(old_name_popt, new_name_popt)









