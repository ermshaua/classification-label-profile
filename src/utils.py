import os
import shutil
import stat
import tempfile
import subprocess

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

    prop_filename = ABS_PATH + f"/../datasets/{dataset}/properties.txt"
    prop_file = []

    with open(prop_filename, 'r') as file:
        for line in file.readlines(): prop_file.append(line.split(","))

    assert len(desc_file) == len(prop_file), "Description and property file have different records."

    df = []

    for idx, (desc_row, prop_row) in enumerate(zip(desc_file, prop_file)):
        if selection is not None and idx not in selection: continue
        assert desc_row[0] == prop_row[0], f"Description and property row {idx} have different records."

        (ts_name, window_size), change_points = desc_row[:2], desc_row[2:]
        labels = prop_row[1:]

        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        path = ABS_PATH + f'/../datasets/{dataset}/'

        if os.path.exists(path + ts_name + ".txt"):
            ts = np.loadtxt(fname=path + ts_name + ".txt", dtype=np.float64)
        else:
            ts = np.load(file=path + "data.npz")[ts_name]

        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]),
                   np.array([int(_) for _ in labels]), ts))

    return pd.DataFrame.from_records(df, columns=["dataset", "window_size", "change_points", "labels", "time_series"])


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


def load_has_datasets(selection=None):
    data_path = ABS_PATH + "/../datasets/has2023_master.csv.zip"

    np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]

    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}

    df_has = pd.read_csv(data_path, converters=converters, compression="zip")

    df = []
    sample_rate = 50

    for _, row in df_has.iterrows():
        if selection is not None and row.ts_challenge_id not in selection: continue
        ts_name = f"{row.group}_subject{row.subject}_routine{row.routine} (id{row.ts_challenge_id})"

        label_mapping = {label: idx for idx, label in enumerate(np.unique(row.activities))}
        labels = np.array([label_mapping[label] for label in row.activities])

        if row.group == "indoor":
            ts = np.hstack((
                row["x-acc"].reshape(-1, 1),
                row["y-acc"].reshape(-1, 1),
                row["z-acc"].reshape(-1, 1),
                row["x-gyro"].reshape(-1, 1),
                row["y-gyro"].reshape(-1, 1),
                row["z-gyro"].reshape(-1, 1),
                row["x-mag"].reshape(-1, 1),
                row["y-mag"].reshape(-1, 1),
                row["z-mag"].reshape(-1, 1)
            ))
        elif row.group == "outdoor":
            ts = np.hstack((
                row["x-acc"].reshape(-1, 1),
                row["y-acc"].reshape(-1, 1),
                row["z-acc"].reshape(-1, 1),
                row["x-mag"].reshape(-1, 1),
                row["y-mag"].reshape(-1, 1),
                row["z-mag"].reshape(-1, 1),
                # row["lat"].reshape(-1, 1),
                # row["lon"].reshape(-1, 1),
                # row["speed"].reshape(-1, 1)
            ))
        else:
            raise ValueError("Unknown group in HAS dataset.")

        df.append((ts_name, sample_rate, row.change_points, labels, ts))

    if selection is None:
        selection = np.arange(df_has.shape[0])

    return pd.DataFrame.from_records(
        df,
        columns=["dataset", "window_size", "change_points", "labels", "time_series"]
    ).iloc[selection, :]


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


def load_train_dataset():
    train_names = [
        'DodgerLoopDay',
        'EEGRat',
        'EEGRat2',
        'FaceFour',
        'GrandMalSeizures2',
        'GreatBarbet1',
        'Herring',
        'InlineSkate',
        'InsectEPG1',
        'MelbournePedestrian',
        'NogunGun',
        'NonInvasiveFetalECGThorax1',
        'ShapesAll',
        'TiltECG',
        'ToeSegmentation1',
        'ToeSegmentation2',
        'Trace',
        'UWaveGestureLibraryY',
        'UWaveGestureLibraryZ',
        'WordSynonyms',
        'Yoga'
    ]

    df = pd.concat([load_datasets("UTSA"), load_tssb_datasets()])
    df = df[df["dataset"].isin(train_names)]

    return df.sort_values(by="dataset")


def normalize_time_series(ts):
    flatten = False

    if ts.ndim == 1:
        ts = ts.reshape(-1,1)
        flatten = True

    for dim in range(ts.shape[1]):
        channel = ts[:,dim]

        # min-max normalize channel
        try:
            channel = np.true_divide(channel - channel.min(), channel.max() - channel.min())
        except FloatingPointError:
            pass

        # interpolate (if missing values are present)
        channel[np.isinf(channel)] = np.nan
        channel = pd.Series(channel).interpolate(limit_direction="both").to_numpy()

        # there are series that still contain NaN values
        channel[np.isnan(channel)] = 0

        ts[:,dim] = channel

    if flatten:
        ts = ts.flatten()

    return ts


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


def create_sliding_window(time_series, window_size, stride=1):
    X = []

    for idx in range(0, time_series.shape[0], stride):
        if idx + window_size <= time_series.shape[0]:
            X.append(time_series[idx:idx + window_size])

    return np.array(X, dtype=time_series.dtype)


def expand_label_sequence(labels, window_size, stride):
    X = []

    for label in labels:
        X.extend([label] * (window_size - (window_size - stride)))

    return np.array(X, dtype=labels.dtype)


def collapse_label_sequence(label_seq):
    labels = []

    for idx in range(1, len(label_seq)):
        if label_seq[idx-1] != label_seq[idx]:
            labels.append(label_seq[idx-1])

        if idx == len(label_seq)-1:
            labels.append(label_seq[idx])

    return np.array(labels)


def extract_cps(label_seq):
    label_diffs = label_seq[:-1] != label_seq[1:]
    return np.arange(label_seq.shape[0] - 1)[label_diffs] + 1