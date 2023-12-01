import matplotlib
import numpy as np

from src.utils import collapse_label_sequence, extract_cps

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme()
sns.set_color_codes()


def plot_time_series(title, time_series, change_points=None, labels=None, file_path=None, font_size=18):
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1,1)

    if change_points is None:
        change_points = np.zeros(0, dtype=int)

    if labels is None:
        labels = np.zeros(change_points.shape[0]+1, dtype=int)

    plt.clf()
    fig, axes = plt.subplots(
        time_series.shape[1],
        sharex=True,
        gridspec_kw={'hspace': .15},
        figsize=(20, time_series.shape[1] * 2)
    )

    if time_series.shape[1] == 1:
        axes = [axes]

    label_colours = {}
    idx = 0

    for activity in labels:
        if activity not in label_colours:
            label_colours[activity] = f"C{idx}"
            idx += 1

    for dim, ax in enumerate(axes):
        ts = time_series[:,dim]

        if len(ts) > 0:
            segments = [0] + change_points.tolist() + [ts.shape[0]]
            for idx in np.arange(0, len(segments) - 1):
                ax.plot(
                    np.arange(segments[idx], segments[idx + 1]),
                    ts[segments[idx]:segments[idx + 1]],
                    c=label_colours[labels[idx]]
                )

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

    axes[0].set_title(title, fontsize=font_size)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")

    return ax


def plot_state_detection(title, time_series, state_seq, change_points=None, labels=None, file_path=None, font_size=18):
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1,1)

    if change_points is None:
        change_points = np.zeros(0, dtype=int)

    if labels is None:
        labels = np.zeros(change_points.shape[0]+1, dtype=int)

    plt.clf()
    fig, axes = plt.subplots(
        time_series.shape[1]+1,
        sharex=True,
        gridspec_kw={'hspace': .15},
        figsize=(20, (time_series.shape[1]+1) * 2)
    )

    label_colours = {}
    state_colours = {}

    idx = 0

    for label in labels:
        if label not in label_colours:
            label_colours[label] = f"C{idx}"
            idx += 1

    for label in np.unique(state_seq):
        state_colours[label] = f"C{idx}"
        idx += 1

    for dim, ax in enumerate(axes):
        if dim < time_series.shape[1]:
            series = time_series[:,dim]
            segments = [0] + change_points.tolist() + [series.shape[0]]
            colors = label_colours
            annotation = labels
        else:
            series = state_seq
            segments = [0] + extract_cps(state_seq).tolist() + [series.shape[0]]
            colors = state_colours
            annotation = collapse_label_sequence(state_seq)


        if len(series) > 0:
            for idx in np.arange(0, len(segments) - 1):
                ax.plot(
                    np.arange(segments[idx], segments[idx + 1]),
                    series[segments[idx]:segments[idx + 1]],
                    c=colors[annotation[idx]]
                )

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

    axes[0].set_title(title, fontsize=font_size)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")

    return ax


