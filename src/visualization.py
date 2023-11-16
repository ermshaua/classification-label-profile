import matplotlib
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme()
sns.set_color_codes()


def plot_time_series(title, time_series, change_points, labels, show=True, file_path=None, font_size=18):
    plt.clf()
    fig, axes = plt.subplots(
        len(time_series),
        sharex=True,
        gridspec_kw={'hspace': .15},
        figsize=(20, len(time_series) * 2)
    )

    label_colours = {}
    idx = 0

    for activity in labels:
        if activity not in label_colours:
            label_colours[activity] = f"C{idx}"
            idx += 1

    for ts, ax in zip(time_series, axes):
        if len(ts) > 0:
            segments = [0] + change_points.tolist() + [ts.shape[0]]
            for idx in np.arange(0, len(segments) - 1):
                ax.plot(
                    np.arange(segments[idx], segments[idx + 1]),
                    ts[segments[idx]:segments[idx + 1]],
                    c=label_colours[labels[idx]]
                )

        # ax.set_ylabel(sensor, fontsize=font_size)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

    axes[0].set_title(title, fontsize=font_size)

    if show is True:
        plt.show()

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")

    return ax