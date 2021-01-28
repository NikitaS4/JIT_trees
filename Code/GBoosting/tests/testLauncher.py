# imports
import argparse
import os, sys
sys.path.append(os.path.abspath('..'))
import JITtrees
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# test cases
from testCases import ModelOptionsHelper, TargetOptionsHelper


class TestHelper:
    @staticmethod
    def test_pipeline(target_options, data_options, model_options, out_options, plot_options):
        # target_options: function + function domain + function representation + function short name
        # data options: data length
        # model options: bins + patience + early stopping delta + ...
        # out options: verbose
        
        # generate dataset
        x_train, y_train, x_valid, y_valid = TestHelper.__generate_data_uniform(target_options, 
            data_options)

        # print experiment conditions
        if out_options['verbose'] >= 3:
            TestHelper.__print_conditions(model_options, data_options, x_train.shape[1])

        # fit model
        def fit_wrapper(JIT_option=model_options['use_jit']):
            model = JITtrees.Boosting(model_options['bins'], model_options['patience'])
            start_time = time.time() # get start time to count the time of execution
            history = model.fit(x_train, y_train, x_valid, y_valid, model_options['tree_count'],
                model_options['tree_depth'], model_options['learning_rate'], model_options['es_delta'],
                model_options['use_jit'])
            exec_time = time.time() - start_time
            if out_options['verbose'] >= 1:
                print("Fit time (" + ("JIT" if JIT_option else "no JIT") + f") = {exec_time} seconds")
            return model, history
        
        history = None
        model = None
        if out_options['compare_jit']:
            fit_wrapper(False)
            model, history = fit_wrapper(True)
        else:
            model, history = fit_wrapper()

        if out_options['verbose'] >= 1:
            print(f"Model fit finished")
        if out_options['verbose'] >= 3:
            print(f"Real tree count: {history.trees_number()}")

        # fit Sklearn model to compare with
        if out_options["sklearn"]:
        #if parsed_flags.compare_sklearn:
            sk_model = HistGradientBoostingRegressor(learning_rate=model_options['learning_rate'],
            max_depth=model_options['tree_depth'], max_iter=model_options['tree_count'])
            sk_model.fit(x_train, y_train)

        # evaluate both models
        preds = model.predict(x_valid)

        if out_options['verbose'] >= 2:
            print("Evaluation:")
            model_mae = mae(y_valid, preds)
            print(f"JITtrees model MAE: {model_mae}")

        if out_options['sklearn'] and out_options['verbose'] >= 2:
        #if parsed_flags.compare_sklearn:
            sk_preds = sk_model.predict(x_valid)    
            sklearn_mae = mae(y_valid, sk_preds)
            print(f"Sklearn model MAE: {sklearn_mae}")
            print(f"Sklearn better {model_mae / sklearn_mae} times")

        # make plots
        if plot_options['need_plots']:
        #if parsed_flags.make_plots:
            #filename = os.path.join('images', 'losses_' + test_name + '.png')
            model_name = TestHelper.__get_model_name(target_options, model_options)
            TestHelper.__plot_losses(history, model_name + "_loss")
            #filename = os.path.join('images', 'preds_' + test_name + '.png')
            #TestHelper.__plot_predictions(data_border, model, sk_model if need_plot_sklearn and parsed_flags.compare_sklearn else None,
            #    target_func, target_repr, filename, plot_errors)
            TestHelper.__plot_predictions(target_options, model, plot_options, 
                model_name + "_pred", sk_model if out_options["sklearn"] else None)

    # "Private" methods - they shouldn't be called directly outside the class

    @staticmethod
    def __generate_data_uniform(target_options, data_options):
    #def __generate_data_uniform(train_cnt, valid_cnt, target_func, data_border):
        train_cnt = data_options["train_cnt"]
        valid_cnt = data_options["valid_cnt"]
        all_cnt = train_cnt + valid_cnt
        shape = (all_cnt, 1)
        data_border = target_options['data_border']
        all_x = np.linspace(-data_border, data_border, num=all_cnt)
        all_y = target_options["func"](all_x)
        all_x = all_x.reshape(shape)
        x_train, x_valid, y_train, y_valid = train_test_split(all_x, all_y,
        test_size=valid_cnt/all_cnt, random_state=42)    
        return x_train, y_train, x_valid, y_valid

    @staticmethod
    def __print_conditions(model_options, data_options, feature_cnt):
    #def __print_conditions(bins, patience, tree_count, learning_rate,
        #tree_depth, train_cnt, valid_cnt, feature_cnt):
        print("Hyperparameters:")
        print(f"Bin count: {model_options['bins']}; Patience: {model_options['patience']}; Tree count: {model_options['tree_count']}")
        print(f"Learning rate: {model_options['learning_rate']}; Tree depth: {model_options['tree_depth']}")
        print("Dataset:")
        print(f"Size: train = {data_options['train_cnt']}; validation = {data_options['valid_cnt']}")
        print(f"Use JIT: {model_options['use_jit']}")
        print(f"Feature count = {feature_cnt}")

    @staticmethod
    def __get_model_name(target_options, model_options):
        jit_str = "JIT" if model_options['use_jit'] else "noJIT"
        return target_options['short_name'] + "_" + jit_str + "_b" + str(model_options['bins']) + "_lr" + str(model_options['learning_rate']) + "_d" + str(model_options['tree_depth'])

    @staticmethod
    def __plot_losses(history, filename, dir="images"):
        plt.plot(history.train_losses(), label='train loss')
        plt.plot(history.valid_losses(), label='valid loss')
        plt.axvline(x=history.trees_number(), color='r', linestyle='--', label='best model')
        plt.legend()
        plt.xlabel('Tree count')
        plt.ylabel(r'$\frac{MSE}{2}$')
        plt.title('Loss dynamics')
        plt.savefig(os.path.join(dir, filename + ".png"))
        plt.show()
        plt.close()

    @staticmethod
    def __plot_predictions(target_options, model, plot_options, filename, sk_model=None, dir="images"):
    #def __plot_predictions(data_border, model, sk_model, target_func, target_repr, filename, plot_errors=False):
        data_border = target_options['data_border']
        x_plot = np.linspace(-data_border, data_border, 1000)
        y_plot = np.array([model.predict([x_plot[i]]) for i in range(x_plot.shape[0])])
        ground_truth = target_options['func'](x_plot)
        plt.plot(x_plot, y_plot, label='prediction')
        plt.plot(x_plot, ground_truth, label=r"Ground truth: " + target_options["repr"])
        if plot_options["plot_sklearn"] and sk_model is not None:
            # need to plot Sklearn model predictions
            y_sk = sk_model.predict(x_plot.reshape(-1, 1))
            plt.plot(x_plot, y_sk, label='Sklearn model', linestyle='dotted')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Model prediction')
        plt.savefig(os.path.join(dir, filename + ".png"))
        plt.show()
        plt.close()

        if plot_options['plot_errors']:
            errors = y_plot - ground_truth
            plt.plot(x_plot, errors)
            plt.xlabel('x')
            plt.ylabel(r'prediction - ground truth')
            plt.title('Prediction residuals')
            plt.savefig(os.path.join(dir, filename + "_err" + ".png"))
            plt.show()
            plt.close()


def check_all_test():
    out_options = {
        "verbose": 1,
        "sklearn": False,
        "compare_jit": True
    }

    plot_options = {
        'need_plots': False,
        'plot_sklearn': False,
        'plot_errors': False
    }

    data_options = {
        'train_cnt': 10000,
        'valid_cnt': 3000
    }

    model_options_list = [
        ModelOptionsHelper.single_tree(),
        ModelOptionsHelper.single_split(),
        ModelOptionsHelper.ensemble()
    ]

    target_options_list = [
        TargetOptionsHelper.linear(),
        TargetOptionsHelper.poly(),
        TargetOptionsHelper.cos(),
        TargetOptionsHelper.sin()
    ]

    tests_cnt = len(target_options_list) * len(model_options_list)
    for i, target_opts in enumerate(target_options_list):
        for j, model_opts in enumerate(model_options_list):
            print(f"Do test {i * len(model_options_list) + j} of {tests_cnt}")
            TestHelper.test_pipeline(target_opts, data_options, model_opts, out_options, 
                plot_options)


def check_fast_test():
    out_options = {
        "verbose": 1,
        "sklearn": False,
        "compare_jit": True
    }

    plot_options = {
        'need_plots': False,
        'plot_sklearn': False,
        'plot_errors': False
    }

    data_options = {
        'train_cnt': 10000,
        'valid_cnt': 1000
    }

    model_options_list = [
        #ModelOptionsHelper.single_tree(),
        #ModelOptionsHelper.single_split(),
        ModelOptionsHelper.ensemble()
    ]

    target_options_list = [
        TargetOptionsHelper.linear(),
        TargetOptionsHelper.poly(),
        TargetOptionsHelper.cos(),
        #TargetOptionsHelper.sin()
    ]

    tests_cnt = len(target_options_list) * len(model_options_list)
    for i, target_opts in enumerate(target_options_list):
        for j, model_opts in enumerate(model_options_list):
            print(f"Do test {i * len(model_options_list) + j} of {tests_cnt}")
            TestHelper.test_pipeline(target_opts, data_options, model_opts, out_options, 
                plot_options)


def entry_point():
    # define command line arguments
    parser = argparse.ArgumentParser(description="testLauncher", add_help=True)

    # Briefly how to launch with CLI:
    # Check all tests in JIT mode (check if model creates correctly): 
    # python testLauncher.py --all
    # Check subset of tests (not all but representative, checks faster):
    # python testLauncher.py --all-fast
    # Check some tests with plots (comment unneeded test cases below):
    # python testLauncher.py -v 3 -p --error-plot --skplot -s -j
    # Compare fit speed of regular and JITed models:
    # python testLauncher.py -v 3 --compare-jit
 
    parser.add_argument('--all', action="store_true", default=False, dest="test_all", help="pass all tests")
    parser.add_argument('--all-fast', action="store_true", default=False, dest="test_fast", help="pass most representative tests")
    parser.add_argument('-v', type=int, default=1, dest="verbose", help="verbose mode (0 - quiet, 3 - the most detailed)")
    parser.add_argument('-p', action="store_true", dest="make_plots", help="make plots")
    parser.add_argument('--error-plot', action="store_true", dest="make_error_plots", help="make plots of errors")
    parser.add_argument('--skplot', action="store_true", dest="plot_sklearn", help="add plot of sklearn model")
    parser.add_argument('-s', action="store_true", dest="compare_sklearn", help="compare with sklearn model")
    parser.add_argument('-j', action="store_true", dest="use_JIT", help="use JIT-compiled trees")
    parser.add_argument('--compare-jit', action="store_false", dest="compare_jit", 
        help="compare reuglar trees and JITed trees (train time)")

    # parse command line arguments
    parsed_flags = parser.parse_args()

    # check main flags
    if parsed_flags.test_all:
        check_all_test()
        return
    
    if parsed_flags.test_fast:
        check_fast_test()
        return

    # create out options from parsed args
    out_options = {
        'verbose': parsed_flags.verbose,
        'sklearn': parsed_flags.compare_sklearn,
        'compare_jit': parsed_flags.compare_jit
    }

    plot_options = {
        'need_plots': parsed_flags.make_plots,
        'plot_sklearn': parsed_flags.plot_sklearn,
        'plot_errors': parsed_flags.make_error_plots
    }

    data_options = {
        'train_cnt': 10000,
        'valid_cnt': 3000
    }

    use_JIT = parsed_flags.use_JIT

    # model options, comment unneeded
    model_options_list = [
        #ModelOptionsHelper.single_tree(use_JIT),
        #ModelOptionsHelper.single_split(use_JIT),
        ModelOptionsHelper.ensemble(use_JIT)
    ]

    # target options, comment unneeded
    target_options_list = [
        #TargetOptionsHelper.linear(),
        #TargetOptionsHelper.poly(),
        TargetOptionsHelper.cos(),
        #TargetOptionsHelper.sin()
    ]

    # launch tests
    tests_cnt = len(target_options_list) * len(model_options_list)
    for i, target_opts in enumerate(target_options_list):
        for j, model_opts in enumerate(model_options_list):
            print(f"Do test {i * len(model_options_list) + j} of {tests_cnt}")
            TestHelper.test_pipeline(target_opts, data_options, model_opts, out_options, 
                plot_options)


if __name__ == "__main__":
    entry_point()
