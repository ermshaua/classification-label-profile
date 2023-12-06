import hashlib
import numpy as np
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.deep_learning import IndividualInceptionClassifier
from aeon.classification.dictionary_based import WEASEL_V2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.shapelet_based._rdst import RDSTClassifier
from claspy.window_size import map_window_size_methods
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
        sample_size = self.window_size
        stride = sample_size // 2

        state_labels = create_state_labels(change_points, labels, time_series.shape[0])

        X, y = [], []

        excl_zone = np.full(time_series.shape[0], fill_value=False, dtype=bool)

        for cp in change_points:
            excl_zone[cp - sample_size // 2 + 1:cp] = True

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
                clf = RocketClassifier(n_jobs=self.n_jobs, random_state=self.random_state)
            elif self.classifier == "weasel":
                clf = WEASEL_V2(random_state=self.random_state, n_jobs=self.n_jobs)
            elif self.classifier == "rdst":
                clf = RDSTClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            elif self.classifier == "dtw":
                clf = KNeighborsTimeSeriesClassifier(distance="dtw", n_jobs=self.n_jobs)
            elif self.classifier == "inception":
                clf = IndividualInceptionClassifier()
            else:
                raise ValueError(f"The classifier {self.classifier} is not supported.")

            y_true[test_idx] = y_test
            y_pred[test_idx] = clf.fit(X_train, y_train).predict(X_test)

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
                W.append(max(1, map_window_size_methods(self.window_size)(time_series[:, dim])))

            if len(W) > 0:
                self.window_size = int(np.mean(W))
            else:
                self.window_size = 10

        if labels is None:
            labels = np.arange(change_points.shape[0] + 1)

        X, y = self._create_dataset(time_series, change_points, labels)
        merged = True

        ignore_cache = set()

        y_true, y_pred = self._cross_val_classifier(*self._subselect_X_y(X, y))

        while merged and np.unique(labels).shape[0] > 1:
            unique_labels = np.unique(labels)

            conf_loss = np.zeros(unique_labels.shape[0], dtype=float)
            conf_index = np.zeros(unique_labels.shape[0], dtype=int)

            # calculate confusions
            for idx, conf in enumerate(confusion_matrix(y_true, y_pred)):
                # drop TPs
                tmp = conf.copy()
                tmp[idx] = 0

                # store most confused label
                conf_index[idx] = np.argmax(tmp)
                conf_loss[idx] = np.max(tmp) / np.sum(conf)

            merged = False

            # merge most confused classes (with descending confusion loss)
            for idx in np.argsort(conf_loss)[::-1]:
                label1, label2 = unique_labels[idx], unique_labels[conf_index[idx]]

                if label1 not in labels or label2 not in labels:
                    continue

                if label1 == label2:
                    continue

                test_idx = np.logical_or(y_true == label1, y_true == label2)
                test_key = hashlib.sha256(test_idx.tobytes()).hexdigest()

                if test_key in ignore_cache:
                    continue

                _y_true, _y_pred = y_true.copy(), y_pred.copy()

                _y_true[_y_true == label2] = label1
                _y_pred[_y_pred == label2] = label1

                if self._classification_gain(y_true, y_pred) > self._classification_gain(_y_true, _y_pred):
                    ignore_cache.add(test_key)
                    continue

                label1, label2 = np.sort([label1, label2])

                labels[labels == label2] = label1
                y[y == label2] = label1

                y_true[y_true == label2] = label1
                y_pred[y_pred == label2] = label1

                merged = True
                break

        # map labels from 0 to n-1
        label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}

        self.labels = np.array([label_mapping[label] for label in labels], dtype=int)
        self.y_true, self.y_pred = y_true, y_pred
        self.cross_val_score = f1_score(self.y_true, self.y_pred, average="macro")

        self.is_fitted = True
        return self

    def random_f1_score(self, y_true):
        labels = np.unique(y_true)

        score = 0

        for label in labels:
            pos_instances = np.sum(y_true == label)
            neg_instances = np.sum(y_true != label)

            tp = pos_instances * pos_instances / y_true.shape[0]
            fn = pos_instances * neg_instances / y_true.shape[0]
            fp = neg_instances * pos_instances / y_true.shape[0]

            pre = tp / (tp + fp)
            re = tp / (tp + fn)

            if pre + re > 0:
                score += 2 * (pre * re) / (pre + re)

        return score / labels.shape[0]

    def _classification_gain(self, y_true, y_pred):
        cross_val_score = f1_score(y_true, y_pred, average="macro")
        rand_score = self.random_f1_score(y_true)
        return cross_val_score - rand_score

    def score(self):
        return self._classification_gain(self.y_true, self.y_pred)

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
