import os

import numpy as np
from numba import prange, njit
from numba.typed.typedlist import List
from stumpy import mass

from claspy.distance import sliding_mean_std
from claspy.nearest_neighbour import _argkmin, _sliding_dot
from claspy.utils import check_input_time_series


@njit(fastmath=True, cache=True)
def _calc_reference_points(n_timepoints, n_references, window_size):
    candidates = [np.arange(window_size, n_timepoints, 2 * window_size)]
    ref_points = []

    while len(ref_points) < n_references and len(candidates) > 0:
        new_candidates = []

        for cand_list in candidates:
            if len(ref_points) == n_references: break

            half = len(cand_list) // 2
            s1, s2 = cand_list[:half], cand_list[half:]

            if len(s1) == 0 and len(s2) > 0:
                ref_points.append(s2[0])
                new_candidates.append(s2[1:])
            elif len(s2) == 0 and len(s1) > 0:
                ref_points.append(s1[-1])
                new_candidates.append(s1[:-1])
            elif len(s1) > 0 and len(s2) > 0:
                ref_points.append(s2[0])
                new_candidates.append(s1)
                new_candidates.append(s2[1:])
            else:
                pass

        candidates = new_candidates

    return np.sort(np.asarray(ref_points, np.int64))


@njit(fastmath=True, cache=True)
def _calc_reference_dist(time_series, window_size, query_point, reference_point):
    dists = mass(
        time_series[query_point:query_point + window_size],
        time_series[max(0, reference_point - window_size):reference_point + window_size]
    )

    return np.min(dists), max(0, reference_point - window_size) + np.argmin(dists)


@njit(fastmath=True, cache=True)
def _calc_reference_ts(time_series, reference_points, window_size):
    ref_ts = np.zeros(2 * window_size * reference_points.shape[0], np.float64)
    ref_points = np.zeros(2 * window_size * reference_points.shape[0], np.int64)

    for idx, ref_point in enumerate(reference_points):
        ref_ts[2 * idx * window_size:(2 * idx + 2) * window_size] = time_series[
                                                                    ref_point - window_size:ref_point + window_size]
        ref_points[2 * idx * window_size:(2 * idx + 2) * window_size] = np.arange(ref_point - window_size,
                                                                                  ref_point + window_size)

    return ref_points, ref_ts


@njit(fastmath=True, cache=True)
def _calc_reference_validity(reference_points, window_size):
    ref_val = np.zeros(2 * window_size * reference_points.shape[0], np.bool_)

    for idx, ref_point in enumerate(reference_points):
        ref_val[2 * idx * window_size:(2 * idx + 1) * window_size] = True

    return ref_val


@njit(fastmath=True, cache=True)
def _knn(time_series, ref_ts, ref_points, ref_val, start, end, window_size, k_neighbours, tcs, dot_ref):
    l1 = len(time_series) - window_size + 1
    l2 = len(ref_ts) - window_size + 1

    exclusion_radius = window_size // 2

    knns = np.zeros(shape=(end - start, len(tcs) * k_neighbours), dtype=np.int64)
    dists = np.zeros(shape=(end - start, len(tcs) * k_neighbours), dtype=np.float64)

    dot_prev = None
    means1, stds1 = sliding_mean_std(time_series, window_size)
    means2, stds2 = sliding_mean_std(ref_ts, window_size)

    for order in range(start, end):
        if order == start:
            dot_rolled = _sliding_dot(time_series[start:start + window_size], ref_ts)
        else:
            dot_rolled = np.roll(dot_prev, 1) \
                         + time_series[order + window_size - 1] * ref_ts[window_size - 1:l2 + window_size] \
                         - time_series[order - 1] * np.roll(ref_ts[:l2], 1)
            dot_rolled[0] = dot_ref[order]

        dist = 2 * window_size * (
                1 - (dot_rolled - window_size * means2 * means1[order]) / (window_size * stds2 * stds1[order]))

        # apply exclusion zone
        lbound, ubound = max(0, order - exclusion_radius), min(order + exclusion_radius + 1, l1)
        dist[np.logical_and(lbound < ref_points[:l2], ref_points[:l2] < ubound)] = np.max(dist)

        # apply validity mask
        dist[np.logical_not(ref_val[:l2])] = np.max(dist)

        for kdx, (lbound, ubound) in enumerate(tcs):
            if order < lbound or order >= ubound: continue
            tmp = dist.copy()

            # apply temporal constraint
            tmp[np.logical_and(ref_points[:l2] < lbound, ubound - window_size + 1 < ref_points[:l2])] = np.max(tmp)
            tc_nn = _argkmin(tmp, k_neighbours)

            dists[order - start, kdx * k_neighbours:(kdx + 1) * k_neighbours] = tmp[tc_nn]
            knns[order - start, kdx * k_neighbours:(kdx + 1) * k_neighbours] = ref_points[tc_nn]

        dot_prev = dot_rolled

    return dists, knns


@njit(fastmath=True, cache=True, parallel=True)
def _parallel_knn(time_series, ref_ts, ref_points, ref_val, window_size, k_neighbours, pranges, tcs):
    knns = np.zeros(shape=(len(time_series) - window_size + 1, len(tcs) * k_neighbours), dtype=np.int64)
    dists = np.zeros(shape=(len(time_series) - window_size + 1, len(tcs) * k_neighbours), dtype=np.float64)

    dot_ref = _sliding_dot(ref_ts[:window_size], time_series)

    for idx in prange(len(pranges)):
        start, end = pranges[idx]

        dists[start:end, :], knns[start:end, :] = _knn(
            time_series,
            ref_ts,
            ref_points,
            ref_val,
            start,
            end,
            window_size,
            k_neighbours,
            tcs,
            dot_ref
        )

    return dists, knns


class KConvergentSubsequenceNeighbours:

    def __init__(self, window_size=10, n_references=100, k_neighbours=3, distance="znormed_euclidean_distance",
                 n_jobs=-1):
        self.window_size = window_size
        self.n_references = n_references
        self.k_neighbours = k_neighbours
        self.distance_name = distance
        self.n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs

    def fit(self, time_series, temporal_constraints=None):
        n_timepoints = time_series.shape[0] - self.window_size + 1

        check_input_time_series(time_series)

        if time_series.shape[0] < self.window_size * self.k_neighbours:
            raise ValueError("Time series must at least have k_neighbours*window_size data points.")

        self.time_series = time_series

        if temporal_constraints is None:
            self.temporal_constraints = np.asarray([(0, time_series.shape[0])], dtype=np.int64)
        else:
            self.temporal_constraints = temporal_constraints

        pranges = List()
        n_jobs = self.n_jobs

        while time_series.shape[0] // n_jobs < self.window_size * self.k_neighbours and n_jobs != 1:
            n_jobs -= 1

        bin_size = time_series.shape[0] // n_jobs

        for idx in range(n_jobs):
            start = idx * bin_size
            end = min((idx + 1) * bin_size, len(time_series) - self.window_size + 1)
            if end > start: pranges.append((start, end))

        ref_points = _calc_reference_points(n_timepoints, self.n_references, self.window_size)
        ref_points, ref_ts = _calc_reference_ts(time_series, ref_points, self.window_size)
        ref_val = _calc_reference_validity(ref_points, self.window_size)

        self.distances, self.offsets = _parallel_knn(
            time_series,
            ref_ts,
            ref_points,
            ref_val,
            self.window_size,
            self.k_neighbours,
            pranges,
            List(self.temporal_constraints),
        )

        return self

    def constrain(self, lbound, ubound):
        if (lbound, ubound) not in self.temporal_constraints:
            raise ValueError(f"({lbound},{ubound}) is not a valid temporal constraint.")

        for idx, tc in enumerate(self.temporal_constraints):
            if tuple(tc) == (lbound, ubound):
                tc_idx = idx

        ts = self.time_series[lbound:ubound]
        distances = self.distances[lbound:ubound - self.window_size + 1,
                    tc_idx * self.k_neighbours:(tc_idx + 1) * self.k_neighbours]
        offsets = self.offsets[lbound:ubound - self.window_size + 1,
                  tc_idx * self.k_neighbours:(tc_idx + 1) * self.k_neighbours] - lbound

        knn = KConvergentSubsequenceNeighbours(
            window_size=self.window_size,
            n_references=self.n_references,
            k_neighbours=self.k_neighbours,
            distance=self.distance_name
        )

        knn.time_series = ts
        knn.temporal_constraints = np.asarray([(0, ts.shape[0])], dtype=np.int64)
        knn.distances, knn.offsets = distances, offsets
        return knn


class SubsequenceReferenceTransform:

    def __init__(self, window_size=10, n_references=10, random_state=2357):
        self.window_size = window_size
        self.n_references = n_references
        self.random_state = random_state

    def fit(self, time_series):
        np.random.seed(self.random_state)
        n_timepoints = time_series.shape[0] - self.window_size + 1

        reference_points = np.random.choice(
            n_timepoints,
            self.n_references,
            replace=False
        )

        X = np.zeros((n_timepoints, self.n_references), np.float64)

        for query_point in range(0, n_timepoints):
            for idx, reference_point in enumerate(reference_points):
                dist, _ = _calc_reference_dist(time_series, self.window_size, query_point, reference_point)
                X[query_point][idx] = dist

        self.X = X
        return self
