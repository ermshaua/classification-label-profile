import numpy as np
from aeon.classification.convolution_based import RocketClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold

from claspy.nearest_neighbour import KSubsequenceNeighbours
from claspy.utils import check_input_time_series
from claspy.window_size import map_window_size_methods
from src.utils import create_state_labels


class CLaP:

    def __init__(self, window_size="suss", n_splits=3, n_jobs=1, random_state=2357):
        self.window_size = window_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_splits = n_splits
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

    def create_dataset(self, time_series, change_points, labels, sample_size=None, stride=None):
        if sample_size is None:
            sample_size = 2 * self.window_size

        if stride is None:
            stride = sample_size // 2

        state_labels = create_state_labels(change_points, labels, time_series.shape[0])

        X, y = [], []

        for idx in range(0, time_series.shape[0] - sample_size + 1, stride):
            X.append([time_series[idx:idx + sample_size]])
            y.append(state_labels[idx])

        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)

    def cross_val_classifier(self, X, y):
        y_true = np.zeros_like(y)
        y_pred = np.zeros_like(y)

        for train_idx, test_idx in KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state).split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            rocket = RocketClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            y_true[test_idx] = y_test
            y_pred[test_idx] = rocket.fit(X_train, y_train).predict(X_test)

        return y_true, y_pred

    def cross_val_knn(self, time_series, y):
        knn = KSubsequenceNeighbours(
            window_size=self.window_size
        ).fit(time_series)

        y_true = y[:-self.window_size + 1]

        n_timepoints, k_neighbours = knn.offsets.shape
        knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

        for i_neighbor in range(k_neighbours):
            neighbours = knn.offsets[:, i_neighbor]
            knn_labels[i_neighbor] = y_true[neighbours]

        y_pred = np.zeros_like(y_true)

        for idx in range(n_timepoints):
            neigh_labels = knn_labels[:, idx]
            u_labels, counts = np.unique(neigh_labels, return_counts=True)
            y_pred[idx] = u_labels[np.argmax(counts)]

        cps = np.array([idx for idx in range(y_true.shape[0] - 1) if y_true[idx] != y_true[idx + 1]])
        labels = np.array(
            [y[idx] for idx in range(y_true.shape[0] - 1) if y_true[idx] != y_true[idx + 1]] + [y_true[-1]])

        for idx, split_idx in enumerate(cps):
            exclusion_zone = np.arange(split_idx - self.window_size, split_idx)
            y_pred[exclusion_zone] = labels[idx + 1]

        return y_true, y_pred

    def fit(self, time_series, change_points, labels=None):
        check_input_time_series(time_series)
        # todo: check change points and labels

        self.time_series = time_series
        self.change_points = change_points

        if isinstance(self.window_size, str):
            self.window_size = max(1, map_window_size_methods(self.window_size)(time_series) // 2)

        if labels is None:
            labels = np.arange(change_points.shape[0] + 1)

        X, y = self.create_dataset(time_series, change_points, labels)
        # y = create_state_labels(change_points, labels, time_series.shape[0])
        merged = True

        while merged and np.unique(labels).shape[0] > 1:
            unique_labels = np.unique(labels)
            y_true, y_pred = self.cross_val_classifier(X, y)
            # y_true, y_pred = self.cross_val_knn(time_series, y)

            conf_matrix = confusion_matrix(y_true, y_pred)

            max_confs = np.zeros(conf_matrix.shape[0], dtype=np.float64)
            arg_max_confs = np.zeros(conf_matrix.shape[0], dtype=np.int64)

            for idx in range(conf_matrix.shape[0]):
                conf_matrix[idx][idx] = 0

                max_confs[idx] = np.max(conf_matrix[idx])
                arg_max_confs[idx] = np.argmax(conf_matrix[idx])

            merged = False

            for idx in np.argsort(max_confs)[::-1]:
                merge_label1 = unique_labels[idx]
                merge_label2 = unique_labels[arg_max_confs[idx]]

                if merge_label1 not in labels or merge_label2 not in labels:
                    continue

                # order merge labels (ascending)
                if merge_label2 < merge_label1:
                    tmp = merge_label1
                    merge_label1 = merge_label2
                    merge_label2 = tmp

                test_idx = np.logical_or(y == merge_label1, y == merge_label2)

                y_true, y_pred = self.cross_val_classifier(X[test_idx], y[test_idx])
                # y_true, y_pred = self.cross_val_knn(time_series[test_idx], y[test_idx])
                score = f1_score(y_true, y_pred, average="macro")

                if score > .75: continue
                # TODO: ranksums(y_pred[:change_point], y_pred[change_point:])

                labels[labels == merge_label2] = merge_label1
                y[y == merge_label2] = merge_label1
                merged = True

        self.labels = labels - labels.min()

        self.is_fitted = True
        return self

    def score(self):
        # todo: compute cross-validation score
        pass

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
                change_points.append(self.change_points[idx - 1])

        return np.asarray(change_points)
