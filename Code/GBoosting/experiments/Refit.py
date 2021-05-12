import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_diabetes, make_regression
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor as SkBoosting
from catboost import Pool, CatBoostRegressor
import traceback
import os, sys
import time
import json

# as the module is created in the upper directory
sys.path.append('..')  
import JITtrees


def save_res(folder, res_name, df_res):
    # save resulting metrics
    target_path = os.path.join(folder, res_name)
    df_res = pd.DataFrame(df_res)
    df_res.to_csv(target_path)


def split_options(model_options):
    # splits model options to the ctor and fit options (for JITtrees model)
    ctor_keys = ['min_bins', 'max_bins', 'patience',
                 'no_early_stopping', 'thread_cnt']
    fit_keys = ['tree_count', 'tree_depth',
                'feature_fold_size', 'learning_rate',
                'early_stopping_delta', 'batch_part',
                'random_state', 'random_batches',
                'regularization_param', 'random_hist_thresholds',
                'remove_regularization_later']
    ctor_options = {}
    fit_options = {}
    for key in model_options.keys():
        if key in ctor_keys:
            ctor_options[key] = model_options[key]
        elif key in fit_keys:
            fit_options[key] = model_options[key]
    return ctor_options, fit_options


def load_dataset(dataset, random_state):
    if dataset == 'boston':
        return load_boston(return_X_y=True)
    elif dataset == 'diabetes':
        return load_diabetes(return_X_y=True)
    elif dataset == 'regr_100':
        return make_regression(n_samples=1000, n_features=100, 
            n_informative=80, n_targets=1, bias=10.0, noise=3.0, shuffle=True, 
            random_state=random_state)
    elif dataset == 'regr_200':
        return make_regression(n_samples=1000, n_features=200, 
            n_informative=150, n_targets=1, bias=10.0, noise=3.0, shuffle=True, 
            random_state=random_state)
    elif dataset == 'winequality':
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
        return x_all, y_all
    elif dataset == 'supercond':
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
    elif dataset == 'stairs':
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
    else:
        raise ValueError('wrong dataset name')


def json_load_utf8(filename):
    with open(filename, encoding='utf-8') as handle:
        loaded = json.load(handle)
    return loaded


def refit_cb(params_file, x_train, x_valid, y_train, y_valid,
    random_seed):
    # read params from file
    params_cb = json_load_utf8(params_file)
    # fit model
    model = CatBoostRegressor(verbose=False, **params_cb)
    start_time = time.time()
    model.fit(X=x_train, y=y_train,
        eval_set=(x_valid, y_valid))
    fit_time = time.time() - start_time
    return model, fit_time


def refit_sk(params_file, x_train, x_valid, y_train, y_valid,
    random_seed):
    # read params from file
    params_sk = json_load_utf8(params_file)
    # fit model
    model = SkBoosting(**params_sk)
    start_time = time.time()
    model.fit(x_train, y_train)
    fit_time = time.time() - start_time
    return model, fit_time


def refit_jt(params_file, x_train, x_valid, y_train, y_valid,
    random_seed):
    # read params from file
    params_jt = json_load_utf8(params_file)
    # fit model
    ctor_options, fit_options = split_options(params_jt)
    model = JITtrees.Boosting(**ctor_options)
    start_time = time.time()
    model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid,
        y_valid=y_valid, **fit_options)
    fit_time = time.time() - start_time
    return model, fit_time


def evaluate_models(models_dict, x_test, y_test, maes, mses):
    for key in models_dict:
        preds = models_dict[key].predict(x_test)
        maes[key].append(mae_score(y_test, preds))
        mses[key].append(mse_score(y_test, preds))


def refit_models(params_file_dict, dataset, folder, test_size=0.2,
    val_size=0.2, random_state=12, fit_baseline=False,
    iterations=1000):
    # get data
    x_all, y_all = load_dataset(dataset, random_state)
    time_dict = {}  # fit times
    maes = {}  # MAE for all models
    mses = {}  # MSE for all models
    refitter_iter = [('CatBoost', refit_cb),
            ('Sklearn', refit_sk),
            ('JITtrees', refit_jt)]
    if fit_baseline:
            refitter_iter.append(('JITtreesBase', refit_jt))
    # init lists in the dicts
    for model_name, cur_refitter in refitter_iter:
        maes[model_name] = []
        mses[model_name] = []
        time_dict[model_name] = []
    # perform experiments many times
    for cur_it in range(iterations):
        # split into [train + validation] and test
        x_tr_val, x_test, y_tr_val, y_test = train_test_split(x_all, y_all, 
            test_size=test_size, random_state=cur_it)
        # split into train and validation
        x_train, x_valid, y_train, y_valid = train_test_split(x_tr_val, y_tr_val,
            test_size=val_size, random_state=cur_it)

        # fit models
        models = {}
        for model_name, cur_refitter in refitter_iter:
            print(f"Fit model {model_name}")
            models[model_name], fit_time = cur_refitter(params_file_dict[model_name],
                x_train, x_valid, y_train, y_valid, random_state)
            time_dict[model_name].append(fit_time)

        # compute metrics
        evaluate_models(models, x_test, y_test, maes, mses)
        print(f"Iteration {cur_it + 1} / {iterations} for {dataset} done")
    # save to file
    save_res(folder, dataset + '_mae_refit.csv', maes)
    save_res(folder, dataset + '_mse_refit.csv', mses)
    save_res(folder, dataset + '_refitTime.csv', time_dict)


def main():
    try:
        FIT_BASELINE = False
        ITERS = 100
        refit_datasets = [
            'boston',
            'diabetes',
            'regr_100',
            'regr_200',
            'winequality',
            'supercond',
            'stairs'
        ]
        folder = 'tuning'
        for cur_dataset in refit_datasets:
            pars_preffix = os.path.join(folder, cur_dataset)
            params_file_dict = {
                'CatBoost': pars_preffix + '_best_pars_catboost.json',
                'Sklearn': pars_preffix + '_best_pars_sklearn.json',
                'JITtrees': pars_preffix + '_best_params.json'
            }
            if FIT_BASELINE:
                params_file_dict['JITtreesBase'] = pars_preffix + '_best_pars_baseline.json'
            refit_models(params_file_dict, cur_dataset, folder,
                test_size=0.2, val_size=0.2, random_state=12,
                fit_baseline=FIT_BASELINE, iterations=ITERS)
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
