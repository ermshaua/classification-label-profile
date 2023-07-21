import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix

from claspy.nearest_neighbour import KSubsequenceNeighbours
from claspy.utils import check_input_time_series
from claspy.window_size import map_window_size_methods
from src.scoring import map_scores
from src.utils import create_state_labels


def cross_val_multilabels(offsets, cps, labels, window_size):
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


class CLaP:

    def __init__(self, window_size="suss", k_neighbours=3, distance="znormed_euclidean_distance", score="f1"):
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

        if isinstance(self.window_size, str):
            self.window_size = max(1, map_window_size_methods(self.window_size)(time_series) // 2)

        knn = KSubsequenceNeighbours(
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            distance=self.distance,
        ).fit(time_series)

        labels = np.arange(change_points.shape[0] + 1)

        y_true, y_pred = cross_val_multilabels(
            knn.offsets,
            change_points,
            labels,
            self.window_size
        )

        score = self.score(y_true, y_pred)

        while np.unique(labels).shape[0] > 1:
            unique_labels = np.unique(labels)

            conf_matrix = confusion_matrix(y_true, y_pred)
            conf_idx, conf = np.zeros(labels.shape[0], np.int64), np.zeros(labels.shape[0], np.float64)

            for idx, c in enumerate(conf_matrix):
                tmp = c.copy()
                tmp[idx] = 0
                kdx = np.argmax(tmp)
                conf_idx[idx], conf[idx] = kdx, c[kdx] / np.sum(c)

            max_conf = np.argmax(conf)
            conf_label1 = unique_labels[max_conf]
            conf_label2 = unique_labels[conf_idx[max_conf]]

            conf_label1_size = conf[max_conf]
            label2_size = np.sum(conf_matrix[conf_idx[max_conf]]) / y_true.shape[0]

            if conf_label1_size / 2 <= label2_size:
                break

            tmp_labels = labels.copy()
            tmp_labels[tmp_labels == conf_label2] = conf_label1

            y_true_tmp, y_pred_tmp = cross_val_multilabels(
                knn.offsets,
                change_points,
                tmp_labels,
                self.window_size
            )

            tmp_score = self.score(y_true_tmp, y_pred_tmp)
            if tmp_score > score:
                score, labels = tmp_score, tmp_labels
                y_true, y_pred = y_true_tmp, y_pred_tmp
            else:
                break

        self.labels = labels - labels.min()

        self.is_fitted = True
        return self
