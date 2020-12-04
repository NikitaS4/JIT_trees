# imports
import os, sys
sys.path.append(os.path.abspath('..'))
import JITtrees
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# test cases
from testCases import lin_5_case, lin_5_neg_case, sin_3_case, sin_2_case, cos_2_case, poly_3_case


def generate_data_uniform(train_cnt, valid_cnt, target_func, data_border):
    all_cnt = train_cnt + valid_cnt
    shape = (all_cnt, 1)
    all_x = np.linspace(-data_border, data_border, num=all_cnt)
    all_y = target_func(all_x)
    all_x = all_x.reshape(shape)
    x_train, x_valid, y_train, y_valid = train_test_split(all_x, all_y,
    test_size=valid_cnt/all_cnt, random_state=42)    
    return x_train, y_train, x_valid, y_valid


def print_conditions(bins, patience, tree_count, learning_rate,
    tree_depth, train_cnt, valid_cnt, feature_cnt):
    print("Hyperparameters:")
    print(f"Bin count: {bins}; Patience: {patience}; Tree count: {tree_count}")
    print(f"Learning rate: {learning_rate}; Tree depth: {tree_depth}")
    print("Dataset:")
    print(f"Size: train = {train_cnt}; validation = {valid_cnt}")
    print(f"Feature count = {feature_cnt}")


def plot_losses(history, filename):
    plt.plot(history.train_losses(), label='train loss')
    plt.plot(history.valid_losses(), label='valid loss')
    plt.axvline(x=history.trees_number(), color='r', linestyle='--', label='best model')
    plt.legend()
    plt.xlabel('Tree count')
    plt.ylabel(r'$\frac{MSE}{2}$')
    plt.title('Loss dynamics')
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_predictions(data_border, model, sk_model, target_func, target_repr, filename):
    x_plot = np.linspace(-data_border, data_border, 1000)
    y_plot = np.array([model.predict([x_plot[i]]) for i in range(x_plot.shape[0])])
    ground_truth = target_func(x_plot)
    plt.plot(x_plot, y_plot, label='prediction')
    plt.plot(x_plot, ground_truth, label=r"Ground truth: " + target_repr)
    if sk_model is not None:
        # don't need to plot Sklearn model predictions
        y_sk = sk_model.predict(x_plot.reshape(-1, 1))
        plt.plot(x_plot, y_sk, label='Sklearn model')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model prediction')
    plt.savefig(filename)
    plt.show()
    plt.close()


def launch_test(case):
    # extract params
    bins = case['bins']
    patience = case['patience']
    es_delta = case['es_delta']
    tree_count = case['tree_count']
    learning_rate = case['learning_rate']
    tree_depth = case['tree_depth']
    data_border = case['data_border']
    train_cnt = case['train_cnt']
    valid_cnt = case['valid_cnt']
    target_func = case['target']
    target_repr = case['target_repr']
    need_plot_sklearn = case['plot_sklearn']
    test_name = case['test_name']

    # generate dataset
    x_train, y_train, x_valid, y_valid = generate_data_uniform(train_cnt,
    valid_cnt, target_func, data_border)

    # print experiment conditions
    print_conditions(bins, patience, tree_count, learning_rate,
    tree_depth, train_cnt, valid_cnt, x_train.shape[1])
    
    # fit model
    model = JITtrees.Boosting(bins, patience)
    history = model.fit(x_train, y_train, x_valid, y_valid, tree_count,
        tree_depth, learning_rate, es_delta)
    print(f"Real tree count: {history.trees_number()}")

    # fit Sklearn model to compare with
    sk_model = HistGradientBoostingRegressor(learning_rate=learning_rate,
    max_depth=tree_depth, max_iter=tree_count)
    sk_model.fit(x_train, y_train)

    # evaluate both models
    preds = np.array([model.predict(x_valid[i,:]) for i in range(x_valid.shape[0])])

    print("Evaluation:")
    model_mae = mae(y_valid, preds)
    print(f"JITtrees model MAE: {model_mae}")
    
    sk_preds = sk_model.predict(x_valid)    
    sklearn_mae = mae(y_valid, sk_preds)
    print(f"Sklearn model MAE: {sklearn_mae}")
    print(f"Sklearn better {model_mae / sklearn_mae} times")

    # make plots
    filename = os.path.join('images', 'losses_' + test_name + '.png')
    plot_losses(history, filename)
    filename = os.path.join('images', 'preds_' + test_name + '.png')
    plot_predictions(data_border, model, sk_model if need_plot_sklearn else None,
        target_func, target_repr, filename)


if __name__ == "__main__":
    # define params to launch test
    test_cases = [#lin_5_case(), lin_5_neg_case(), sin_2_case(), sin_3_case(), cos_2_case(),
        poly_3_case()]
    
    for case in test_cases:
        launch_test(case)
