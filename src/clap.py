import numpy as np
from numba import njit
from sklearn.exceptions import NotFittedError

from claspy.nearest_neighbour import KSubsequenceNeighbours
from claspy.scoring import map_scores
from claspy.utils import check_input_time_series
from src.utils import create_segmentation_labels


def cross_val_multilabels(offsets, cps, labels, window_size):
    n_timepoints, k_neighbours = offsets.shape

    y_true = create_segmentation_labels(cps, labels, n_timepoints)
    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = offsets[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    y_pred = np.zeros_like(y_true)

    for idx in range(n_timepoints):
        neigh_labels = knn_labels[:, idx]
        u_labels, counts = np.unique(neigh_labels, return_counts=True)
        y_pred[idx] = u_labels[np.argmax(counts)]

    for split_idx in cps:
        exclusion_zone = np.arange(split_idx - window_size, split_idx)
        y_pred[exclusion_zone] = 1

    return y_true, y_pred


class CLaP:

    def __init__(self, window_size=10, k_neighbours=3, distance="znormed_euclidean_distance", score="roc_auc"):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.distance = distance
        self.score_name = score
        self.score = map_scores(score)
        self.is_fitted = False

    def _check_is_fitted(self):
        """
        Checks if the CLaP object is fitted.

        Raises
        ------
        NotFittedError
            If the CLaP object is not fitted.

        Returns
        -------
        None
        """
        if not self.is_fitted:
            raise NotFittedError("CLaP object is not fitted yet. Please fit the object before using this method.")

    def fit(self, time_series, change_points):
        check_input_time_series(time_series)
        # todo: check change points

        labels = np.zeros(shape=change_points.shape[0]+1, dtype=np.int64)

        if change_points.shape[0] > 0:
            labels[1] = 1

        segments = change_points.tolist() + [time_series.shape[0]]

        last_label = np.max(labels)

        for idx in range(2, len(segments)):
            seg_start, seg_end = segments[idx-1], segments[idx]

            tmp_series = time_series[:seg_end]
            tmp_points = change_points[:idx]
            tmp_labels = np.arange(0, last_label+2)
            tmp_scores = np.zeros(shape=tmp_labels.shape[0], dtype=np.float64)

            knn = KSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                distance=self.distance,
            ).fit(tmp_series)

            for kdx, tmp_label in enumerate(tmp_labels):
                label_config = np.concatenate((labels[:idx], [tmp_label]))

                y_true, y_pred = cross_val_multilabels(
                    knn.offsets,
                    tmp_points,
                    label_config,
                    self.window_size
                )

                tmp_scores[kdx] = self.score(y_true[seg_start:], y_pred[seg_start:])

            labels[idx] = tmp_labels[np.argmax(tmp_scores)]
            last_label = np.max(labels)

        self.labels = labels

        self.is_fitted = True
        return self
