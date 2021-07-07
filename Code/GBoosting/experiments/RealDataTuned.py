import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GridSearchCV
from sklearn.datasets import load_boston, load_diabetes, make_regression
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor as SkBoosting
from catboost import Pool, CatBoostRegressor
import traceback
import os, sys
import time
import json

# as the module is created in the upper directory
sys.path.append('..')  
import regbm


def split_options(model_options):
    # splits model options to the ctor and fit options (for regbm model)
    ctor_keys = ['min_bins', 'max_bins', 'patience',
                 'no_early_stopping', 'thread_cnt']
    fit_keys = ['tree_count', 'tree_depth',
                'feature_fold_size', 'learning_rate',
                'early_stopping_delta', 'batch_part',
                'random_state', 'batch_strategy',
                'regularization_param', 'random_hist_thresholds',
                'remove_regularization_later',
                'spoil_split_scores']
    ctor_options = {}
    fit_options = {}
    for key in model_options.keys():
        if key in ctor_keys:
            ctor_options[key] = model_options[key]
        elif key in fit_keys:
            fit_options[key] = model_options[key]
    return ctor_options, fit_options


def get_fit_steps(options_grid):
    # for KFold CV
    return int(np.prod(np.array([len(param_list) for param_list in options_grid.values()])))


def tune_regbm(x_tr_val, y_tr_val, options_grid, random_state=12):
    keys_list = list(options_grid.keys())
    options_count = len(keys_list)
    cur_idx_each_option = [0] * options_count
    cur_prop_to_change = 0
    # the dictionary is the seed for the data frame
    # init the dictionary
    tuning_df = { keys_list[i]: [] for i in range(options_count) }
    tuning_df['MAE'] = []
    tuning_df['time'] = []
    cv_fit_number = get_fit_steps(options_grid)
    print(f"The number of iterations to tune JIT trees model: {cv_fit_number}")
    split_cnt_limit = 5  # K < 5 (in K Fold) => #valid < 0.2 #tr_val
    repeats_cnt = cv_fit_number // split_cnt_limit + 1  # repeats * splits >= cv_fit_number
    kf_cv = RepeatedKFold(n_splits=split_cnt_limit, n_repeats=repeats_cnt, random_state=random_state)  # init CV
    # iterate over splits, but stop when the whole grid will be studied
    iters_gone = 0
    for train_idxs, valid_idxs in kf_cv.split(x_tr_val):
        print(f"Current tuning iteration: {iters_gone + 1} / {cv_fit_number}")
        # get current options
        model_options = {keys_list[i]: options_grid[keys_list[i]][cur_idx_each_option[i]] for i in range(options_count)}
        
        # get train and test sets
        x_train, x_valid = x_tr_val[train_idxs], x_tr_val[valid_idxs]
        y_train, y_valid = y_tr_val[train_idxs], y_tr_val[valid_idxs]
        
        # fit
        ctor_options, fit_options = split_options(model_options)
        model = regbm.Boosting(**ctor_options)
        start_time = time.time()
        history = model.fit(x_train=x_train, y_train=y_train, 
            x_valid=x_valid, y_valid=y_valid, **fit_options)
        exec_time = time.time() - start_time
        
        # evaluate
        preds = model.predict(x_valid)
        if np.isnan(preds).any() or (preds == np.inf).any():
            mae = np.inf
        else:
            try:
                mae = mae_score(y_valid, preds)
            except Exception as m:
                mae = np.inf
        
        # add to the dictionary (to save in the data frame later)
        for key in keys_list:
            tuning_df[key].append(model_options[key])
        tuning_df['MAE'].append(mae)
        tuning_df['time'].append(exec_time)
        
        # update options' indexes
        while cur_prop_to_change < options_count and cur_idx_each_option[cur_prop_to_change] + 1 >= len(options_grid[keys_list[cur_prop_to_change]]):
            # find the next changable option
            cur_prop_to_change += 1
        if cur_prop_to_change >= options_count:
            # we have seen all the options, can finish
            break
        for prev_prop in range(cur_prop_to_change):
            # set all previous options to 0
            cur_idx_each_option[prev_prop] = 0
        cur_idx_each_option[cur_prop_to_change] += 1  # increment the current option
        # reduce index to the start (lexicographic order)
        cur_prop_to_change = 0
        # update iterations counter
        iters_gone += 1
        if iters_gone >= cv_fit_number:
            break  # can finish tuning

    # return the resulting data frame
    tuning_df = pd.DataFrame(tuning_df)  # convert to DF
    best_idx = tuning_df['MAE'].idxmin()  # get minimum by MAE score
    best_params = tuning_df.iloc[[best_idx]].to_dict()
    for key in best_params.keys():
        # convert dictionaries to params (forget indexes)
        best_params[key] = list(best_params[key].values())[0]
    # return the data frame (protocol) and the best parameters dictionary (with mae and exec time)
    return tuning_df, best_params


def tune_CatBoost(x_tr_val, y_tr_val, options_grid, random_state=12):
    model = CatBoostRegressor()
    grid_search_result = model.grid_search(options_grid, 
                                           X=x_tr_val, 
                                           y=y_tr_val,
                                           cv=get_fit_steps(options_grid),
                                           refit=True,
                                           plot=False,
                                           verbose=False,
                                           partition_random_seed=random_state)
    return grid_search_result['params'], model  # return the best parameters and the final model 


def tune_Sklearn(x_tr_val, y_tr_val, options_grid):
    sk_boosting = SkBoosting()
    model = GridSearchCV(sk_boosting, options_grid, refit=True)
    model.fit(x_tr_val, y_tr_val)
    return model.best_params_, model


def regbm_tuned_mae(x_tr_val, y_tr_val, x_test, y_test,
    best_params, preds_dict):
    ctor_options, fit_options = split_options(best_params)
    model = regbm.Boosting(**ctor_options)
    history = model.fit(x_train=x_tr_val, y_train=y_tr_val,
        x_valid=x_test, y_valid=y_test, **fit_options)
    preds = model.predict(x_test)
    if (preds == np.inf).any():  # filter outliers
        return None, None
    mae = mae_score(y_test, preds)
    preds_dict["regbm"] = preds
    return mae, np.std(np.abs(preds - y_test))


def Sklearn_tuned_mae(model, x_test, y_test, preds_dict):
    preds = model.predict(x_test)
    mae = mae_score(y_test, preds)
    preds_dict["sklearn"] = preds
    return mae, np.std(np.abs(preds - y_test))


def CatBoost_tuned_mae(model, x_test, y_test, preds_dict):
    preds = model.predict(x_test)
    mae = mae_score(y_test, preds)
    preds_dict["catboost"] = preds
    return mae, np.std(np.abs(preds - y_test))


def save_res(folder, res_name, prot_name, df_res, df_prot):
    # save resulting metrics
    target_path = os.path.join(folder, res_name)
    df_res = pd.DataFrame(df_res)
    df_res.to_csv(target_path)
    
    # save protocol
    target_path = os.path.join(folder, prot_name)
    df_prot.to_csv(target_path)


def save_best_params(best_params_dict, folder, res_name):
    target_path = os.path.join(folder, res_name)
    with open(target_path, 'w') as file:
        json.dump(best_params_dict, file)


def save_preds_gt(preds_dict, folder, res_name):
    os.makedirs(folder, exist_ok=True)  # create dir if it doesn't exist
    target_path = os.path.join(folder, res_name)
    preds_df = pd.DataFrame(preds_dict)
    preds_df.to_csv(target_path)


def tune_dataset(cb_grid, sk_grid, jt_grid,
    dataset_loader, folder, dataset_name, random_state=12):
    # get data
    x_all, y_all = dataset_loader()
    # split into [train + validation] and test
    x_tr_val, x_test, y_tr_val, y_test = train_test_split(x_all, y_all, 
        test_size=0.2, random_state=random_state)
    
    # dict to form the data frame later
    df_res = {"Models": ["CatBoost", "Sklearn", "regbm"],
              "MAE": [],
              "std": []}

    # dict with predictions and the ground truth
    preds_dict = {'ground_truth': y_test}

    # fit models
    # CatBoost
    print(f"Tune CatBoost model")
    cb_best_params, model = tune_CatBoost(x_tr_val, y_tr_val, cb_grid, random_state)
    cb_mae, cb_sd = CatBoost_tuned_mae(model, x_test, y_test, preds_dict)
    df_res["MAE"].append(cb_mae)
    df_res["std"].append(cb_sd)

    # Sklearn
    print(f"Tune Sklearn model")
    sk_best_params, model = tune_Sklearn(x_tr_val, y_tr_val, sk_grid)
    sk_mae, sk_sd = Sklearn_tuned_mae(model, x_test, y_test, preds_dict)
    df_res["MAE"].append(sk_mae)
    df_res["std"].append(sk_sd)

    # regbm
    print(f"Tune regbm model")
    jt_prot, best_params = tune_regbm(x_tr_val, y_tr_val, jt_grid, random_state)
    jt_mae, jt_sd = regbm_tuned_mae(x_tr_val, y_tr_val, x_test,
        y_test, best_params, preds_dict)
    if jt_mae is not None:
        df_res["MAE"].append(jt_mae)
    if jt_sd is not None:
        df_res["std"].append(jt_sd)

    # save results
    save_res(folder, dataset_name + '.csv', dataset_name + '_prot.csv', df_res, jt_prot)
    # save the best parameters
    save_best_params(best_params, folder, dataset_name + '_best_params.json')
    save_best_params(sk_best_params, folder, dataset_name + '_best_pars_sklearn.json')
    save_best_params(cb_best_params, folder, dataset_name + '_best_pars_catboost.json')
    # save predictions (to be able to make plots/boxplots/etc.)
    save_preds_gt(preds_dict, os.path.join(folder, 'preds_df'),
        dataset_name + '_preds.csv')


def tune_boston(folder, random_state=12):
    # CatBoost
    CatBoost_grid = {
        "iterations": [250, 500],
        "learning_rate": [0.1, 0.2],
        "depth": [2, 4, 6],
        "random_state": [random_state],
        "feature_border_type": ["GreedyLogSum"]
    }

    # Sklearn
    Sklearn_grid = {
        'learning_rate': [0.1, 0.2],
        'max_iter': [250, 500],
        'max_depth': [2, 4, 6]
    }

    # regbm
    regbm_grid = {
        'min_bins': [16, 32, 64],
        'max_bins': [256],
        'no_early_stopping': [False],
        'patience': [4],
        'tree_count': [500, 1000, 2000],
        'tree_depth': [4, 6, 7, 10],
        'feature_fold_size': [1.0],
        'learning_rate': [0.15, 0.2],
        'regularization_param': [0.1],
        'es_delta': [1e-5],
        'batch_part': [0.4, 0.6, 0.8],
        'batch_strategy': [2],
        'random_hist_thresholds': [True],
        'remove_regularization_later': [False],
        'spoil_split_scores': [False, True],
        'thread_cnt': [1]
    }

    tuning_params = {
        'cb_grid': CatBoost_grid, 
        'sk_grid': Sklearn_grid,
        'jt_grid': regbm_grid,
        'dataset_loader': lambda: load_boston(return_X_y=True),
        'folder': folder,
        'dataset_name': 'boston', 
        'random_state': 12
    }
    tune_dataset(**tuning_params)


def tune_diabetes(folder, random_state=12):
    # CatBoost
    CatBoost_grid = {
        "iterations": [250, 500],
        "learning_rate": [0.1, 0.2],
        "depth": [2, 4, 6],
        "random_state": [random_state],
        "feature_border_type": ["GreedyLogSum"]
    }

    # Sklearn
    Sklearn_grid = {
        'learning_rate': [0.1, 0.2],
        'max_iter': [250, 500],
        'max_depth': [2, 4, 6]
    }

    # regbm
    regbm_grid = {
        'min_bins': [16, 32, 64],
        'max_bins': [256],
        'no_early_stopping': [False],
        'patience': [5],
        'tree_count': [500, 1000, 2000],
        'tree_depth': [4, 5, 7],
        'feature_fold_size': [1.0],
        'learning_rate': [0.15],
        'regularization_param': [0.05, 0.1, 0.15],
        'es_delta': [1e-5],
        'batch_part': [0.6, 0.8],
        'batch_strategy': [2],
        'random_hist_thresholds': [True],
        'remove_regularization_later': [False],
        'spoil_split_scores': [False, True],
        'thread_cnt': [1]
    }

    tuning_params = {
        'cb_grid': CatBoost_grid, 
        'sk_grid': Sklearn_grid,
        'jt_grid': regbm_grid,
        'dataset_loader': lambda: load_diabetes(return_X_y=True),
        'folder': folder,
        'dataset_name': 'diabetes', 
        'random_state': 12
    }
    tune_dataset(**tuning_params)


def tune_regression_100(folder, random_state=12):
    # CatBoost
    CatBoost_grid = {
        "iterations": [250, 500],
        "learning_rate": [0.1, 0.2],
        "depth": [2, 4, 6],
        "random_state": [random_state],
        "feature_border_type": ["GreedyLogSum"]
    }

    # Sklearn
    Sklearn_grid = {
        'learning_rate': [0.1, 0.2],
        'max_iter': [250, 500],
        'max_depth': [2, 4, 6]
    }

    # regbm
    regbm_grid = {
        'min_bins': [8, 16, 32, 64],
        'max_bins': [256],
        'no_early_stopping': [False],
        'patience': [4],
        'tree_count': [200, 300, 500, 1000, 2000],
        'tree_depth': [3, 4, 5, 8],
        'feature_fold_size': [1.0],
        'learning_rate': [0.1, 0.2, 0.4, 0.6],
        'regularization_param': [0.6, 0.7, 0.8],
        'es_delta': [1e-5],
        'batch_part': [1],
        'batch_strategy': [0],
        'random_hist_thresholds': [True],
        'remove_regularization_later': [True],
        'spoil_split_scores': [False, True],
        'thread_cnt': [1]
    }

    tuning_params = {
        'cb_grid': CatBoost_grid, 
        'sk_grid': Sklearn_grid,
        'jt_grid': regbm_grid,
        'dataset_loader': lambda: make_regression(n_samples=1000, n_features=100, 
            n_informative=80, n_targets=1, bias=10.0, noise=3.0, shuffle=True, 
            random_state=random_state),
        'folder': folder,
        'dataset_name': 'regr_100', 
        'random_state': 12
    }
    tune_dataset(**tuning_params)


def tune_regression_200(folder, random_state=12):
    # CatBoost
    CatBoost_grid = {
        "iterations": [250, 500],
        "learning_rate": [0.1, 0.2],
        "depth": [2, 4, 6],
        "random_state": [random_state],
        "feature_border_type": ["GreedyLogSum"]
    }

    # Sklearn
    Sklearn_grid = {
        'learning_rate': [0.1, 0.2],
        'max_iter': [250, 500],
        'max_depth': [2, 4, 6]
    }

    # regbm
    regbm_grid = {
        'min_bins': [8, 16, 32, 64],
        'max_bins': [256],
        'no_early_stopping': [False],
        'patience': [4],
        'tree_count': [300, 1000, 2000],
        'tree_depth': [4, 5, 8],
        'feature_fold_size': [1.0],
        'learning_rate': [0.06, 0.1],
        'regularization_param': [0.17, 0.18, 0.19],
        'es_delta': [1e-5],
        'batch_part': [1],
        'batch_strategy': [0],
        'random_hist_thresholds': [True],
        'remove_regularization_later': [True],
        'spoil_split_scores': [False, True],
        'thread_cnt': [1]
    }
    tuning_params = {
        'cb_grid': CatBoost_grid, 
        'sk_grid': Sklearn_grid,
        'jt_grid': regbm_grid,
        'dataset_loader': lambda: make_regression(n_samples=1000, n_features=200, 
            n_informative=150, n_targets=1, bias=10.0, noise=3.0, shuffle=True, 
            random_state=random_state),
        'folder': folder,
        'dataset_name': 'regr_200', 
        'random_state': 12
    }
    tune_dataset(**tuning_params)


def tune_winequality(folder, random_state=12):
    # a bit more complex function to get the data
    def dataset_loader():
        data_dir = os.path.join('datasets', 'winequality')
        data_csv = 'winequality-white.csv'
        all_data = pd.read_csv(os.path.join(data_dir, data_csv))
        # split into target and features
        label_name = 'quality'
        labels_df = all_data[label_name]  # target df
        features_df = all_data.drop(label_name, axis=1)  # features df
        # convert to numpy arrays
        y_all = labels_df.to_numpy()
        x_all = features_df.to_numpy()
        print(f"dataset size: {y_all.shape[0]}")
        print(f"feature count: {x_all.shape[1]}")
        return x_all, y_all

    # CatBoost
    CatBoost_grid = {
        "iterations": [250, 500],
        "learning_rate": [0.1, 0.2],
        "depth": [2, 4, 7, 8],
        "random_state": [random_state],
        "feature_border_type": ["GreedyLogSum"]
    }

    # Sklearn
    Sklearn_grid = {
        'learning_rate': [0.1, 0.2],
        'max_iter': [250, 500],
        'max_depth': [2, 4, 7, 8]
    }

    # regbm
    regbm_grid = {
        'min_bins': [16, 32, 64],
        'max_bins': [256],
        'no_early_stopping': [False],
        'patience': [5],
        'tree_count': [500, 1000, 2000],
        'tree_depth': [3, 5, 7, 8],
        'feature_fold_size': [1.0],
        'learning_rate': [0.12, 0.15, 0.18],
        'regularization_param': [0.1, 0.12, 0.15],
        'es_delta': [1e-6],
        'batch_part': [0.6, 0.8],
        'batch_strategy': [2],
        'random_hist_thresholds': [True],
        'remove_regularization_later': [False],
        'spoil_split_scores': [False, True],
        'thread_cnt': [1]
    }

    tuning_params = {
        'cb_grid': CatBoost_grid, 
        'sk_grid': Sklearn_grid,
        'jt_grid': regbm_grid,
        'dataset_loader': dataset_loader,
        'folder': folder,
        'dataset_name': 'winequality', 
        'random_state': 12
    }
    tune_dataset(**tuning_params)


def tune_supercond(folder, random_state=12):
    # a bit more complex function to get the data
    def dataset_loader():
        data_dir = os.path.join('datasets', 'superconduct')
        data_csv = 'train.csv'
        all_data = pd.read_csv(os.path.join(data_dir, data_csv))
        # split into target and features
        label_name = 'critical_temp'
        labels_df = all_data[label_name]  # target df
        features_df = all_data.drop(label_name, axis=1)  # featrues df
        # convert to numpy arrays
        y_all = labels_df.to_numpy()
        x_all = features_df.to_numpy()
        return x_all, y_all

    # CatBoost
    CatBoost_grid = {
        "iterations": [250, 500],
        "learning_rate": [0.1, 0.2],
        "depth": [2, 4, 7, 8],
        "random_state": [random_state],
        "feature_border_type": ["GreedyLogSum"]
    }

    # Sklearn
    Sklearn_grid = {
        'learning_rate': [0.1, 0.2],
        'max_iter': [250, 500],
        'max_depth': [2, 4, 7, 8]
    }

    # regbm
    regbm_grid = {
        'min_bins': [128, 256],
        'max_bins': [256],
        'no_early_stopping': [False],
        'patience': [4],
        'tree_count': [500, 1000, 2000],
        'tree_depth': [3, 4, 7, 8],
        'feature_fold_size': [1.0],
        'learning_rate': [0.1, 0.15, 0.2],
        'regularization_param': [0, 0.1, 1],
        'es_delta': [1e-6],
        'batch_part': [0.6],
        'batch_strategy': [0],
        'random_hist_thresholds': [True],
        'remove_regularization_later': [True],
        'spoil_split_scores': [False, True],
        'thread_cnt': [1]
    }

    tuning_params = {
        'cb_grid': CatBoost_grid, 
        'sk_grid': Sklearn_grid,
        'jt_grid': regbm_grid,
        'dataset_loader': dataset_loader,
        'folder': folder,
        'dataset_name': 'supercond', 
        'random_state': 12
    }
    tune_dataset(**tuning_params)


def tune_stairs(folder, random_state=12):
    # a bit more complex function to get the data
    def dataset_loader():
        data_dir = 'datasets'
        data_csv = 'stairs.csv'
        all_data = pd.read_csv(os.path.join(data_dir, data_csv))
        # split into target and features
        label_name = 'y'
        labels_df = all_data[label_name]  # target df
        features_df = all_data.drop(label_name, axis=1)  # featrues df
        # convert to numpy arrays
        y_all = labels_df.to_numpy()
        x_all = features_df.to_numpy()
        return x_all, y_all

    # CatBoost
    CatBoost_grid = {
        "iterations": [300, 500],
        "learning_rate": [0.1, 0.2],
        "depth": [2, 4, 7, 8],
        "random_state": [random_state],
        "feature_border_type": ["GreedyLogSum"]
    }

    # Sklearn
    Sklearn_grid = {
        'learning_rate': [0.1, 0.2],
        'max_iter': [300, 500],
        'max_depth': [2, 4, 7, 8]
    }

    # regbm
    regbm_grid = {
        'min_bins': [16, 32, 256],
        'max_bins': [256],
        'no_early_stopping': [False],
        'patience': [4],
        'tree_count': [1000, 2000],
        'tree_depth': [4, 5, 8],
        'feature_fold_size': [1.0],
        'learning_rate': [0.1, 0.12],
        'regularization_param': [0, 0.12, 0.13, 0.14, 0.145],
        'es_delta': [1e-6],
        'batch_part': [0.6, 0.8],
        'batch_strategy': [2],
        'random_hist_thresholds': [True],
        'remove_regularization_later': [False],
        'spoil_split_scores': [False],
        'thread_cnt': [1]
    }

    tuning_params = {
        'cb_grid': CatBoost_grid, 
        'sk_grid': Sklearn_grid,
        'jt_grid': regbm_grid,
        'dataset_loader': dataset_loader,
        'folder': folder,
        'dataset_name': 'stairs', 
        'random_state': 12
    }
    tune_dataset(**tuning_params)


def tune_regression_1(folder, random_state=12):
    # CatBoost
    CatBoost_grid = {
        "iterations": [250, 500],
        "learning_rate": [0.1, 0.2],
        "depth": [2, 4, 6],
        "random_state": [random_state],
        "feature_border_type": ["GreedyLogSum"]
    }

    # Sklearn
    Sklearn_grid = {
        'learning_rate': [0.1, 0.2],
        'max_iter': [250, 500],
        'max_depth': [2, 4, 6]
    }

    # regbm
    regbm_grid = {
        'min_bins': [16, 64, 256],
        'max_bins': [256],
        'no_early_stopping': [False],
        'patience': [4],
        'tree_count': [500, 1000, 2000],
        'tree_depth': [4, 6, 8],
        'feature_fold_size': [1.0],
        'learning_rate': [0.08, 0.1, 0.12],
        'regularization_param': [0.1, 0.12, 0.15],
        'es_delta': [1e-5],
        'batch_part': [0.6, 0.8],
        'batch_strategy': [2],
        'random_hist_thresholds': [True],
        'remove_regularization_later': [False],
        'spoil_split_scores': [False, True],
        'thread_cnt': [1]
    }

    tuning_params = {
        'cb_grid': CatBoost_grid, 
        'sk_grid': Sklearn_grid,
        'jt_grid': regbm_grid,
        'dataset_loader': lambda: make_regression(n_samples=5000, n_features=1, 
            n_informative=1, n_targets=1, bias=0.0, noise=0.2, shuffle=True, 
            random_state=random_state),
        'folder': folder,
        'dataset_name': 'regr_1', 
        'random_state': 12
    }
    tune_dataset(**tuning_params)


def main():
    try:
        for cur_tuner in [
                          tune_boston,
                          tune_diabetes,
                          tune_regression_100,
                          tune_regression_200,
                          tune_winequality,
                          tune_supercond,
                          tune_stairs,
                          tune_regression_1
                        ]:
            cur_tuner('tuning', 12)
    except Exception as ex:
        print("An exception was raised")
        print(ex)
        # print traceback
        try:  # need try-finally to delete ex_info
            ex_info = sys.exc_info()
        finally:
            traceback.print_exception(*ex_info)
            del ex_info
    finally:
        print("Finish")


if __name__ == "__main__":
    main()
