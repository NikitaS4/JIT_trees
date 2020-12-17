import os, sys
sys.path.append(os.path.abspath('..'))
import JITtrees
import numpy as np
from matplotlib import pyplot as plt


from sklearn.metrics import mean_absolute_error as mae
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split


def sin_func(x):
    return np.sin(x)


def generate_data_random(train_cnt, valid_cnt, target_func):
    all_cnt = train_cnt + valid_cnt
    var_cnt = 1
    shape = (all_cnt, var_cnt)
    all_x = np.random.rand(*shape)  # x in [0; 1)
    all_x = (all_x - 0.5) * 6  # x in [-3; 3)
    all_y = target_func(all_x[:,0])
    train_x = all_x[:train_cnt,:]
    train_y = all_y[:train_cnt]
    valid_x = all_x[train_cnt:,:]
    valid_y = all_y[train_cnt:]
    return train_x, train_y, valid_x, valid_y


def generate_data_uniform(train_cnt, valid_cnt, target_func):
    all_cnt = train_cnt + valid_cnt
    var_cnt = 1
    shape = (all_cnt, var_cnt)
    all_x = np.linspace(-3, 3, num=all_cnt)
    all_y = target_func(all_x)
    all_x = all_x.reshape(shape)
    valid_idx = range(0, all_cnt, train_cnt // valid_cnt)
    x_train, x_valid, y_train, y_valid = train_test_split(all_x, all_y,
    test_size=valid_cnt/all_cnt, random_state=42)
    return x_train, y_train, x_valid, y_valid


if __name__ == "__main__":
    # hyperparameters
    bins = 256
    patience = 6
    tree_count = 1000  # use stop with patience
    learning_rate = 0.3
    tree_depth = 4  # use 2 features per tree

    # dataset parameters
    train_cnt = 100000
    valid_cnt = 1000
    x_train, y_train, x_valid, y_valid = generate_data_uniform(train_cnt, 
        valid_cnt, sin_func)

    # pre-summary
    print("Hyperparameters:")
    print(f"Bin count: {bins}; Patience: {patience}; Tree count: {tree_count}")
    print(f"Learning rate: {learning_rate}; Tree depth: {tree_depth}")
    print("Dataset:")
    print(f"Size: train = {train_cnt}; validation = {valid_cnt}")
    print(f"Feature count = {x_train.shape[1]}")

    model = JITtrees.Boosting(bins, patience)
    history = model.fit(x_train, y_train, x_valid, y_valid,
        tree_count, tree_depth, learning_rate)
    
    # summary
    print("Model has been fit")
    print(f"Trees built: {history.trees_number()}")

    # estimation
    preds = np.array([model.predict(x_valid[i,:]) for i in range(x_valid.shape[0])])
    tr_preds = np.array([model.predict(x_train[i,:]) for i in range(x_train.shape[0])])

    print("Estimate:")
    print(f"MAE: {mae(y_valid, preds)}")

    # fit sklearn model to compare with
    model2 = HistGradientBoostingRegressor(learning_rate=learning_rate, max_depth=tree_depth,
    max_iter=tree_count)
    model2.fit(x_train, y_train)
    sk_preds = model2.predict(x_valid)
    sk_train_preds = model2.predict(x_train)
    print(f"Sklearn model MAE: {mae(y_valid, sk_preds)}")

    plt.plot(history.train_losses(), label='train loss')
    plt.plot(history.valid_losses(), label='valid loss')
    plt.axvline(x=history.trees_number(), color='r', linestyle='--', label='best model')
    plt.legend(loc='upper right')
    plt.xlabel('Tree count')
    plt.ylabel(r'$\frac{MSE}{2}$')
    plt.title('Loss dynamics')    
    plt.savefig(os.path.join('images', 'lossSingle.png'))
    plt.show()
    plt.close()

    x_plot = np.linspace(-3, 3, 1000)
    y_plot = np.array([model.predict([x_plot[i]]) for i in range(x_plot.shape[0])])
    ground_truth = np.sin(x_plot)
    plt.plot(x_plot, y_plot, label='prediction')
    plt.plot(x_plot, ground_truth, label='Ground truth = sin(x)')
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model prediction')    
    plt.savefig(os.path.join('images', 'predSingle.png'))
    plt.show()
    plt.close()
