import os

from numba import njit

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd


def load_datasets(dataset, selection=None):
    desc_filename = ABS_PATH + f"/../datasets/{dataset}/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []

    for idx, row in enumerate(desc_file):
        if selection is not None and idx not in selection: continue
        (ts_name, window_size), change_points = row[:2], row[2:]
        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        path = ABS_PATH + f'/../datasets/{dataset}/'

        if os.path.exists(path + ts_name + ".txt"):
            ts = np.loadtxt(fname=path + ts_name + ".txt", dtype=np.float64)
        else:
            ts = np.load(file=path + "data.npz")[ts_name]

        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change_points", "time_series"])


def load_tssb_datasets(names=None):
    desc_filename = os.path.join(ABS_PATH, "../datasets/TSSB", "desc.txt")
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines():
            line = line.split(",")

            if names is None or line[0] in names:
                desc_file.append(line)

    prop_filename = os.path.join(ABS_PATH, "../datasets/TSSB", "properties.txt")
    prop_file = []

    with open(prop_filename, 'r') as file:
        for line in file.readlines():
            line = line.split(",")

            ds_name, interpretable, label_cut, resample_rate, labels = line[0], bool(line[1]), int(line[2]), int(
                line[3]), line[4:]
            labels = [int(l.replace("\n", "")) // (label_cut + 1) for l in labels]

            if names is None or ds_name in names:
                prop_file.append((ds_name, label_cut, resample_rate, labels))

    df = []

    for desc_row, prop_row in zip(desc_file, prop_file):
        (ts_name, window_size), change_points = desc_row[:2], desc_row[2:]
        labels = prop_row[3]

        ts = np.loadtxt(fname=os.path.join(ABS_PATH, "../datasets/TSSB", ts_name + '.txt'), dtype=np.float64)
        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), np.array(labels), ts))

    return pd.DataFrame.from_records(df, columns=["dataset", "window_size", "change_points", "labels", "time_series"])


def load_mosad_datasets():
    cp_filename = ABS_PATH + "/../datasets/MOSAD/change_points.txt"
    cp_file = []

    with open(cp_filename, 'r') as file:
        for line in file.readlines(): cp_file.append(line.split(","))

    activity_filename = ABS_PATH + "/../datasets/MOSAD/activities.txt"
    activities = dict()

    with open(activity_filename, 'r') as file:
        for line in file.readlines():
            line = line.split(",")
            routine, motions = line[0], line[1:]
            activities[routine] = [motion.replace("\n", "") for motion in motions]

    ts_filename = ABS_PATH + "/../datasets/MOSAD/data.npz"
    T = np.load(file=ts_filename)

    df = []

    for row in cp_file:
        (ts_name, sample_rate), change_points = row[:2], row[2:]
        routine, subject, sensor = ts_name.split("_")
        ts = T[ts_name]

        df.append((ts_name, int(routine[-1]), int(subject[-1]), sensor, int(sample_rate),
                   np.array([int(_) for _ in change_points]), np.array(activities[routine[-1]]), ts))

    return pd.DataFrame.from_records(df,
                                     columns=["dataset", "routine", "subject", "sensor", "sample_rate", "change_points",
                                              "activities", "time_series"])


@njit(fastmath=True, cache=True)
def create_state_labels(cps, labels, ts_len):
    seg_labels = np.zeros(shape=ts_len, dtype=np.int64)

    segments = np.concatenate((
        np.array([0]),
        cps,
        np.array([ts_len])
    ))

    for idx in range(1, len(segments)):
        seg_start, seg_end = segments[idx - 1], segments[idx]
        seg_labels[seg_start:seg_end] = labels[idx - 1]

    return seg_labels


def cross_val_knn(offsets, cps, labels, window_size):
    n_timepoints, k_neighbours = offsets.shape

    y_true = create_state_labels(cps, labels, n_timepoints)
    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = offsets[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    y_pred = np.zeros_like(y_true)

    for idx in range(n_timepoints):
        neigh_labels = knn_labels[:, idx]
        u_labels, counts = np.unique(neigh_labels, return_counts=True)
        y_pred[idx] = u_labels[np.argmax(counts)]

    for idx, split_idx in enumerate(cps):
        exclusion_zone = np.arange(split_idx - window_size, split_idx)
        y_pred[exclusion_zone] = labels[idx + 1]

    return y_true, y_pred


def create_sliding_window(time_series, window_size):
    shape = time_series.shape[:-1] + (time_series.shape[-1] - window_size + 1, window_size)
    strides = time_series.strides + (time_series.strides[-1],)
    return np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)


class AeonTransformerWrapper(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        df = pd.DataFrame()
        df['dim_0'] = [pd.Series(ts) for ts in X]
        self.estimator.fit(df, y)
        return self

    def transform(self, X):
        df = pd.DataFrame()
        df['dim_0'] = [pd.Series(ts) for ts in X]
        return self.estimator.transform(df).to_numpy()