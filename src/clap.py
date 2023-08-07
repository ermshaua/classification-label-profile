import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix, f1_score

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

    def __init__(self, window_size="suss", k_neighbours=3, distance="znormed_euclidean_distance", n_jobs=-1):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.distance = distance
        self.n_jobs = n_jobs
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

    def _compute_conf_labels(self, labels, conf_matrix, conf, conf_idx, n_samples):
        unique_labels = np.unique(labels)
        conf_ind = np.argsort(conf)[::-1]

        for max_conf in conf_ind:
            conf_label1 = unique_labels[max_conf]
            conf_label2 = unique_labels[conf_idx[max_conf]]

            conf_label1_size = conf[max_conf]
            label2_size = np.sum(conf_matrix[conf_idx[max_conf]]) / n_samples

            if conf_label1_size / 2 <= label2_size:
                continue

            return conf_label1, conf_label2

        return None, None

    def fit(self, time_series, change_points, knn=None, labels=None):
        check_input_time_series(time_series)
        # todo: check change points and labels

        self.time_series = time_series
        self.change_points = change_points

        if isinstance(self.window_size, str):
            self.window_size = max(1, map_window_size_methods(self.window_size)(time_series) // 2)

        if knn is None:
            self.knn = KSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                distance=self.distance,
                n_jobs=self.n_jobs
            ).fit(time_series)
        else:
            self.knn = knn

        if labels is None:
            labels = np.arange(change_points.shape[0] + 1)

        while np.unique(labels).shape[0] > 1:
            y_true, y_pred = cross_val_multilabels(
                self.knn.offsets,
                change_points,
                labels,
                self.window_size
            )

            conf_matrix = confusion_matrix(y_true, y_pred)
            conf_idx, conf = np.zeros(conf_matrix.shape[0], np.int64), np.zeros(conf_matrix.shape[0], np.float64)

            for idx, c in enumerate(conf_matrix):
                tmp = c.copy()
                tmp[idx] = 0
                kdx = np.argmax(tmp)
                conf_idx[idx], conf[idx] = kdx, c[kdx] / np.sum(c)

            conf_label1, conf_label2 = self._compute_conf_labels(labels, conf_matrix, conf, conf_idx, y_true.shape[0])

            if None in (conf_label1, conf_label2):
                break

            labels[labels == conf_label2] = conf_label1

        self.labels = labels - labels.min()

        self.is_fitted = True
        return self

    def score(self):
        y_true, y_pred = cross_val_multilabels(
            self.knn.offsets,
            self.change_points,
            self.labels,
            self.window_size
        )

        unique_labels = np.unique(self.labels)

        score = 0

        for label in unique_labels:
            pos_size = np.sum(y_true == label)
            tp_size = np.sum(np.logical_and(y_true == label, y_pred == label))

            prior = pos_size / y_true.shape[0]
            posterior = tp_size / pos_size

            score += posterior - prior

        return score / unique_labels.shape[0]

    def get_segment_labels(self):
        labels = [self.labels[0]]

        for idx in np.arange(1, self.labels.shape[0]):
            if labels[-1] != self.labels[idx]:
                labels.append(self.labels[idx])

        return np.asarray(labels)

    def get_change_points(self):
        labels = [self.labels[0]]
        change_points = []

        for idx in np.arange(1, self.labels.shape[0]):
            if labels[-1] != self.labels[idx]:
                labels.append(self.labels[idx])
                change_points.append(self.change_points[idx-1])

        return np.asarray(change_points)