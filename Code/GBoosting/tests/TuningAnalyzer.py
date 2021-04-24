import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error as mae_score


def make_boxplot(jt_errs, jt_no_improv, sk_errs, cb_errs,
    dataset_name, folder):
    target_file = os.path.join(folder, dataset_name) + '_boxplot.png'
    all_errs = [jt_errs, jt_no_improv, sk_errs, cb_errs] if not(jt_no_improv is None) else [jt_errs, sk_errs, cb_errs]
    xlabels = ['JIT improved', 'JIT base', 'Sklearn', 'CatBoost'] if not(jt_no_improv is None) else ['JIT trees', 'Sklearn', 'CatBoost']
    fig, ax = plt.subplots(1, 1)
    plt.boxplot(all_errs)
    plt.title(f'Absolute errors for {dataset_name}')
    plt.setp(ax, xticks=[y + 1 for y in range(len(all_errs))],
         xticklabels=xlabels)
    plt.ylabel('Absolute error')
    plt.savefig(target_file)
    plt.close()

def inspect_experiment(csv_file, folder, dataset_name, add_baseline):
    csv_full_name = os.path.join(folder, csv_file)
    df = pd.read_csv(csv_full_name, sep=',')
    all_preds = df.to_numpy()
    gt_col, cb_col, sk_col, jt_col, jt_base = 1, 2, 3, 4, 5
    gt = all_preds[:, gt_col]  # the ground truth
    # compute absolute errors
    cb_errs = np.abs(gt - all_preds[:, cb_col])  # CatBoost
    sk_errs = np.abs(gt - all_preds[:, sk_col])  # Scikit-learn
    jt_errs = np.abs(gt - all_preds[:, jt_col])  # JIT_trees (with improvements)
    jt_no_improv = np.abs(gt - all_preds[:, jt_base]) if add_baseline else None # JIT trees
    # create boxplots
    make_boxplot(jt_errs, jt_no_improv, sk_errs, cb_errs,
        dataset_name, folder)


def main():
    ADD_BASELINE = False
    preds_dir = os.path.join('tuning', 'preds_df')
    for filename in os.listdir(preds_dir):
        if filename.endswith('.csv'):
            # extract dataset name
            csv_suffix = r'_preds_refit.csv'
            pattern = r'(.*)' + csv_suffix
            matched = re.match(pattern, filename)
            if matched:
                dataset_name = re.match(pattern, filename).group(1)
                # make boxplot
                inspect_experiment(filename, preds_dir, dataset_name,
                    ADD_BASELINE)


if __name__ == "__main__":
    main()
