import numpy as np
import pandas as pd
import os, sys
from matplotlib import pyplot as plt


def generate_stairs(data_len, stair_cnt=512, x_step=15):
    x_low, x_up = 0, x_step * stair_cnt
    y_step = x_step * 3
    x_all = np.random.uniform(low=x_low, high=x_up, size=data_len)
    y_all = np.floor(x_all / x_step) * y_step
    return x_all, y_all


def generate_shelves(data_len, shelf_cnt=512, x_step=15):
    x_low, x_up = 0, x_step * shelf_cnt
    y_step = x_step * 3
    x_all = np.random.uniform(low=x_low, high=x_up, size=data_len)
    shelves = np.random.randint(low=0, high=shelf_cnt, size=shelf_cnt)
    y_all = shelves[np.floor(x_all / x_step)] * y_step
    return x_all, y_all


def emit_dataset(data_len, filename, generator, show=False):
    x_all, y_all = generator(data_len)
    df = pd.DataFrame(dict(x=x_all, y=y_all))
    df.to_csv(filename, index=False)
    plt.scatter(x_all, y_all)
    plt.show()
    plt.close()


def main():
    data_len = 5000
    show = False
    stairs_fname = os.path.join('datasets', 'stairs.csv')
    shelves_fname = os.path.join('datasets', 'shelves.csv')
    emit_dataset(data_len, stairs_fname, generate_stairs, show)  # stairs
    emit_dataset(data_len, shelves_fname, generate_shelves, show)  # shelves


if __name__ == "__main__":
    main()
