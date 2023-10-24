import os

import numpy as np
from numba import njit, prange
from numba.typed.typedlist import List
from sklearn.exceptions import NotFittedError

from claspy.nearest_neighbour import cross_val_labels
from claspy.scoring import map_scores
from claspy.utils import check_input_time_series, check_excl_radius
from claspy.validation import map_validation_tests
from src.contractable_neighbour import KContractableSubsequenceNeighbours


@njit(fastmath=True, cache=False)
def _profile(offsets, start, end, window_size, score):
    profile = np.full(shape=end - start, fill_value=-np.inf, dtype=np.float64)

    for split_idx in range(start, end):
        y_true, y_pred = cross_val_labels(offsets, split_idx, window_size)
        profile[split_idx - start] = score(y_true, y_pred)

    return profile


@njit(fastmath=True, cache=True, parallel=True)
def _parallel_profile(offsets, pranges, window_size, score):
    profile = np.full(shape=offsets.shape[0], fill_value=-np.inf, dtype=np.float64)

    for idx in prange(len(pranges)):
        start, end = pranges[idx]
        profile[start:end] = _profile(offsets, start, end, window_size, score)

    return profile


class ClaSP:

    def __init__(self, window_size=10, n_references=100, k_neighbours=3, distance="znormed_euclidean_distance",
                 score="roc_auc",
                 excl_radius=5, n_jobs=-1):
        self.window_size = window_size
        self.n_references = n_references
        self.k_neighbours = k_neighbours
        self.distance = distance
        self.score_name = score
        self.score = map_scores(score)
        self.excl_radius = excl_radius
        self.n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        self.is_fitted = False

        check_excl_radius(k_neighbours, excl_radius)

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise NotFittedError("ClaSP object is not fitted yet. Please fit the object before using this method.")

    def fit(self, time_series, knn=None):
        check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius
        self.lbound, self.ubound = 0, time_series.shape[0]

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series

        if knn is None:
            self.knn = KContractableSubsequenceNeighbours(
                window_size=self.window_size,
                n_references=self.n_references,
                k_neighbours=self.k_neighbours,
                distance=self.distance,
                n_jobs=self.n_jobs
            ).fit(time_series)
        else:
            self.knn = knn

        pranges = List()
        n_jobs = self.n_jobs

        while self.knn.offsets.shape[0] // n_jobs < self.min_seg_size and n_jobs != 1:
            n_jobs -= 1

        bin_size = self.knn.offsets.shape[0] // n_jobs

        for idx in range(n_jobs):
            start = max(idx * bin_size, self.min_seg_size)
            end = min((idx + 1) * bin_size, self.knn.offsets.shape[0] - self.min_seg_size + self.window_size)
            if end > start: pranges.append((start, end))

        self.profile = _parallel_profile(self.knn.offsets, pranges, self.window_size, self.score)

        self.is_fitted = True
        return self

    def transform(self):
        self._check_is_fitted()
        return self.profile

    def fit_transform(self, time_series, knn=None):
        return self.fit(time_series, knn).transform()

    def split(self, sparse=True, validation="significance_test", threshold=1e-15):
        self._check_is_fitted()
        cp = np.argmax(self.profile)

        if validation is not None:
            validation_test = map_validation_tests(validation)
            if not validation_test(self, cp, threshold): return None

        if sparse is True:
            return cp

        return self.time_series[:cp], self.time_series[cp:]


class ClaSPEnsemble(ClaSP):

    def __init__(self, n_estimators=10, window_size=10, n_references=100, k_neighbours=3, distance="znormed_euclidean_distance",
                 score="roc_auc", early_stopping=True, excl_radius=5, n_jobs=-1, random_state=2357):
        super().__init__(window_size, n_references, k_neighbours, distance, score, excl_radius, n_jobs)
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.random_state = random_state

    def _calculate_temporal_constraints(self):
        tcs = [(0, self.time_series.shape[0])]
        np.random.seed(self.random_state)

        while len(tcs) < self.n_estimators and self.time_series.shape[0] > 3 * self.min_seg_size:
            lbound, area = np.random.choice(self.time_series.shape[0], 2, replace=True)

            if self.time_series.shape[0] - lbound < area:
                area = self.time_series.shape[0] - lbound

            ubound = lbound + area
            if ubound - lbound < 2 * self.min_seg_size: continue
            tcs.append((lbound, ubound))

        return np.asarray(sorted(tcs, key=lambda tc: tc[1] - tc[0], reverse=True), dtype=np.int64)

    def fit(self, time_series, knn=None, validation="significance_test", threshold=1e-15):
        check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series
        tcs = self._calculate_temporal_constraints()

        if knn is None:
            knn = KContractableSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                distance=self.distance,
                n_jobs=self.n_jobs
            ).fit(time_series, temporal_constraints=tcs)

        best_score, best_tc, best_clasp = -np.inf, None, None

        for idx, (lbound, ubound) in enumerate(tcs):
            clasp = ClaSP(
                window_size=self.window_size,
                n_references=self.n_references,
                k_neighbours=self.k_neighbours,
                score=self.score_name,
                excl_radius=self.excl_radius,
                n_jobs=self.n_jobs
            ).fit(time_series[lbound:ubound], knn=knn.constrain(lbound, ubound))

            clasp.profile = (clasp.profile + (ubound - lbound) / time_series.shape[0]) / 2

            if clasp.profile.max() > best_score or best_clasp is None and idx == tcs.shape[0] - 1:
                best_score = clasp.profile.max()
                best_tc = (lbound, ubound)
                best_clasp = clasp
            else:
                if self.early_stopping is True and best_clasp is not None: break

            if self.early_stopping is True and best_clasp is not None and best_clasp.split(validation=validation,
                                                                                           threshold=threshold) is not None:
                break

        self.knn = best_clasp.knn
        self.lbound, self.ubound = best_tc
        self.profile = np.full(shape=time_series.shape[0] - self.window_size + 1, fill_value=-np.inf, dtype=np.float64)
        self.profile[self.lbound:self.ubound - self.window_size + 1] = best_clasp.profile

        self.is_fitted = True
        return self
