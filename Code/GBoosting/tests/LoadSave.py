import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error as mae_score
import os, sys

# as the module is created in the upper directory
sys.path.append('..')  
import JITtrees


def main():
    rand_state = 12
    cpt_file = os.path.join('checkpoints', 'test.txt')
    cpt_file2 = os.path.join('checkpoints', 'test2.txt')
    # make dataset
    x_all, y_all = make_regression(n_samples=1000, n_features=3,
        n_informative=3, n_targets=1, shuffle=True,
        random_state=rand_state)
    # split
    x_tr, x_test, y_tr, y_test = train_test_split(x_all, y_all,
        test_size=0.2, random_state=rand_state)
    # fit
    model = JITtrees.Boosting(min_bins=256, max_bins=256,
        no_early_stopping=True, thread_cnt=1)
    model.fit(x_train=x_tr, y_train=y_tr, x_valid=x_test,
        y_valid=y_test, tree_count=2, tree_depth=2,
        feature_fold_size=1.0, learning_rate=0.5,
        random_state=rand_state)
    preds = model.predict(x_test)
    mae = mae_score(y_test, preds)
    print(f"MAE: {mae}")
    model.save_model(cpt_file)
    loaded = JITtrees.Boosting(filename=cpt_file,
        thread_cnt=1)
    loaded.save_model(cpt_file2)
    preds = loaded.predict(x_test)
    mae_new = mae_score(y_test, preds)
    print(f"Saved & loaded MAE: {mae}")
    print(f"Test passed: {np.isclose(mae, mae_new)}")
    print("Finish")


if __name__ == "__main__":
    main()
