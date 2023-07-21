import os

from numba import njit

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd


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
