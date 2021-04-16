import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error as mae_score


def make_boxplot(jt_errs, sk_errs, cb_errs, dataset_name, folder):
    target_file = os.path.join(folder, dataset_name) + '_boxplot.png'
    all_errs = [jt_errs, sk_errs, cb_errs]
    fig, ax = plt.subplots(1, 1)
    plt.boxplot(all_errs)
    plt.title(f'Absolute errors for {dataset_name}')
    plt.setp(ax, xticks=[y + 1 for y in range(len(all_errs))],
         xticklabels=['JIT trees', 'Sklearn', 'CatBoost'])
    plt.ylabel('Absolute error')
    plt.savefig(target_file)
    plt.close()

def inspect_experiment(csv_file, folder, dataset_name):
    csv_full_name = os.path.join(folder, csv_file)
    df = pd.read_csv(csv_full_name, sep=',')
    all_preds = df.to_numpy()
    gt_col, cb_col, sk_col, jt_col = 1, 2, 3, 4
    gt = all_preds[:, gt_col]  # the ground truth
    # compute absolute errors
    cb_errs = np.abs(gt - all_preds[:, cb_col])  # CatBoost
    sk_errs = np.abs(gt - all_preds[:, sk_col])  # Scikit-learn
    jt_errs = np.abs(gt - all_preds[:, jt_col])  # JIT_trees
    # create boxplots
    make_boxplot(jt_errs, sk_errs, cb_errs, dataset_name, folder)


def main():
    preds_dir = os.path.join('tuning', 'preds_df')
    for filename in os.listdir(preds_dir):
        if filename.endswith('.csv'):
            # extract dataset name
            csv_suffix = r'_preds.csv'
            pattern = r'(.*)' + csv_suffix
            dataset_name = re.match(pattern, filename).group(1)
            # make boxplot
            inspect_experiment(filename, preds_dir, dataset_name)


if __name__ == "__main__":
    main()
