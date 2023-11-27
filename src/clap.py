import hashlib

import numpy as np
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.dictionary_based import WEASEL_V2
from aeon.classification.shapelet_based._rdst import RDSTClassifier
from claspy.nearest_neighbour import KSubsequenceNeighbours
from claspy.window_size import map_window_size_methods
from scipy.stats import ranksums
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold

from src.utils import create_state_labels


class CLaP:

    def __init__(self, window_size="suss", classifier="rocket", n_splits=5, n_jobs=1, sample_size=1_000,
                 random_state=2357):
        self.window_size = window_size
        self.classifier = classifier
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.sample_size = sample_size

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

    def _create_dataset(self, time_series, change_points, labels):
        sample_size = 3 * self.window_size
        stride = sample_size // 2

        state_labels = create_state_labels(change_points, labels, time_series.shape[0])

        X, y = [], []

        excl_zone = np.full(time_series.shape[0], fill_value=False, dtype=bool)

        # Currently reducing performance
        for cp in change_points:
            excl_zone[cp - sample_size + 1:cp] = True

        for idx in range(0, time_series.shape[0] - sample_size + 1, stride):
            if not excl_zone[idx]:
                window = time_series[idx:idx + sample_size].T
                if time_series.shape[1] == 1: window = window.flatten()

                X.append(window)
                y.append(state_labels[idx])

        return np.array(X, dtype=float), np.array(y, dtype=int)

    def _cross_val_classifier(self, X, y):
        y_true = np.zeros_like(y)
        y_pred = np.zeros_like(y)

        n_splits = min(X.shape[0], self.n_splits)

        if n_splits < 2:
            return np.copy(y), np.copy([y])

        for train_idx, test_idx in KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state).split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if self.classifier == "rocket":
                clf = RocketClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            elif self.classifier == "weasel":
                clf = WEASEL_V2(random_state=self.random_state, n_jobs=self.n_jobs)
            elif self.classifier == "rdst":
                clf = RDSTClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            else:
                raise ValueError(f"The classifier {self.classifier} is not supported.")

            y_true[test_idx] = y_test
            y_pred[test_idx] = clf.fit(X_train, y_train).predict(X_test)

        return y_true, y_pred

    def _cross_val_knn(self, time_series, y):
        knn = KSubsequenceNeighbours(
            window_size=self.window_size
        ).fit(time_series)

        y_true = y[:-self.window_size + 1]

        n_timepoints, k_neighbours = knn.offsets.shape
        knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=int)

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
            exclusion_zone = np.arange(split_idx - self.window_size, split_idx)  # + 1
            y_pred[exclusion_zone] = labels[idx + 1]

        return y_true, y_pred

    def _subselect_X_y(self, X, y):
        np.random.seed(self.random_state)
        labels = np.unique(y)

        X_sel, y_sel = [], []

        for label in labels:
            args = y == label
            X_label, y_label = X[args], y[args]

            if X_label.shape[0] > self.sample_size:
                rand_idx = np.random.choice(np.arange(y.shape[0])[args], self.sample_size, replace=False)
                X_label, y_label = X[rand_idx], y[rand_idx]

            X_sel.extend(X_label)
            y_sel.extend(y_label)

        X_sel, y_sel = np.array(X_sel, dtype=float), np.array(y_sel, dtype=int)

        # randomize order
        args = np.random.choice(X_sel.shape[0], X_sel.shape[0], replace=False)
        return X_sel[args], y_sel[args]

    def estimate_significance_level(self, y_true, y_pred, mask1, mask2, n_iter=100):
        p_vals = []

        for _ in range(n_iter):
            rand1_idx = np.random.choice(np.arange(y_true.shape[0])[np.logical_or(mask1, mask2)], np.sum(mask1), replace=False)
            rand2_idx = np.random.choice(np.arange(y_true.shape[0])[np.logical_or(mask1, mask2)], np.sum(mask2), replace=False)

            # create samples
            x1_rand, x2_rand = y_pred[rand1_idx], y_pred[rand2_idx]

            if x1_rand.shape[0] == 0 or x2_rand.shape[0] == 0:
                continue

            # resampling (currently decreases performance)
            x1_rand = x1_rand[np.random.choice(x1_rand.shape[0], self.sample_size // 2, replace=True)]
            x2_rand = x2_rand[np.random.choice(x2_rand.shape[0], self.sample_size // 2, replace=True)]

            _, p_rand = ranksums(x1_rand, x2_rand)
            p_vals.append(p_rand)

        if len(p_vals) > 0:
            return np.mean(p_vals)

        return 0.

    def _test_ranksums(self,  y_true, y_pred, mask1, mask2):
        alpha = self.estimate_significance_level(y_true, y_pred, mask1, mask2)

        # create samples
        x1, x2 = y_pred[mask1], y_pred[mask2]

        if x1.shape[0] == 0 or x2.shape[0] == 0:
            return True

        # resampling (currently decreases performance)
        x1 = x1[np.random.choice(x1.shape[0], self.sample_size // 2, replace=True)]
        x2 = x2[np.random.choice(x2.shape[0], self.sample_size // 2, replace=True)]

        _, p = ranksums(x1, x2)
        return p < alpha

    def fit(self, time_series, change_points, labels=None):
        np.random.seed(self.random_state)
        # todo: check ts, change points and labels

        if time_series.ndim == 1:
            # make ts multi-dimensional
            time_series = time_series.reshape(-1, 1)

        self.time_series = time_series
        self.change_points = change_points

        W = []

        if isinstance(self.window_size, str):
            for dim in range(time_series.shape[1]):
                W.append(max(1, map_window_size_methods(self.window_size)(time_series[:, dim]) // 2))

            if len(W) > 0:
                self.window_size = int(np.mean(W))
            else:
                self.window_size = 10

        if labels is None:
            labels = np.arange(change_points.shape[0] + 1)

        X, y = self._create_dataset(time_series, change_points, labels)
        # y = create_state_labels(change_points, labels, time_series.shape[0])
        merged = True

        ignore_cache = set()

        y_true, y_pred = self._cross_val_classifier(*self._subselect_X_y(X, y))
        # y_true, y_pred = self.cross_val_knn(time_series, y)

        while merged and np.unique(labels).shape[0] > 1:
            unique_labels = np.unique(labels)

            conf_matrix = confusion_matrix(y_true, y_pred).astype(float)

            max_confs = np.zeros(conf_matrix.shape[0], dtype=float)
            arg_max_confs = np.zeros(conf_matrix.shape[0], dtype=int)

            for idx in range(conf_matrix.shape[0]):
                conf_matrix[idx][idx] = 0

                # normalize confusion matrix
                if np.sum(conf_matrix[idx]) > 0:
                    conf_matrix[idx] /= np.sum(conf_matrix[idx])

                # store most confused classes
                max_confs[idx] = np.max(conf_matrix[idx])
                arg_max_confs[idx] = np.argmax(conf_matrix[idx])

            merged = False

            # merge most confused class (with descending priority)
            for idx in np.argsort(max_confs)[::-1]:
                merge_label1 = unique_labels[idx]
                merge_label2 = unique_labels[arg_max_confs[idx]]

                if merge_label1 not in labels or merge_label2 not in labels:
                    continue

                if merge_label1 == merge_label2:
                    continue

                # order merge labels (ascending)
                if merge_label2 < merge_label1:
                    tmp = merge_label1
                    merge_label1 = merge_label2
                    merge_label2 = tmp

                test_idx = np.logical_or(y_true == merge_label1, y_true == merge_label2)
                test_idx_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()

                if test_idx_hash in ignore_cache:
                    continue

                # create label masks
                mask1, mask2 = y_true == merge_label1, y_true == merge_label2

                # test if labels should be merged
                if self._test_ranksums(y_true, y_pred, mask1, mask2):
                    ignore_cache.add(test_idx_hash)
                    continue

                labels[labels == merge_label2] = merge_label1
                y[y == merge_label2] = merge_label1

                y_true[y_true == merge_label2] = merge_label1
                y_pred[y_pred == merge_label2] = merge_label1

                merged = True
                break

        self.labels = labels - labels.min()
        self.y_true, self.y_pred = y_true, y_pred
        self.cross_val_score = f1_score(self.y_true, self.y_pred, average="macro")

        self.is_fitted = True
        return self

    def score(self, n_iter=100):
        np.random.seed(self.random_state)
        scores = []

        for idx in range(n_iter):
            y_pred = self.y_pred.copy()
            np.random.shuffle(y_pred)
            rand_score = f1_score(self.y_true, y_pred, average="macro")
            scores.append(self.cross_val_score - rand_score)

        return np.mean(scores)


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