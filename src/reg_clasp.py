import os

import numpy as np
from numba import njit, prange
from numba.typed.typedlist import List
from sklearn.exceptions import NotFittedError

from claspy.scoring import map_scores
from claspy.utils import check_input_time_series
from src.regression import SubsequenceRegressor, cross_val_labels
from src.validation import map_validation_tests


@njit(fastmath=True, cache=False)
def _profile(offsets, start, end, window_size, score):
    """
    Computes the classification score profile given nearest neighbour offsets.

    Parameters
    ----------
    offsets : np.ndarray
        An array of shape (n_timepoints, k_neighbors) containing the offsets of the k nearest
        neighbors for each timepoint.
    start : int
        The first index to consider (inclusive).
    end : int
        The last index to consider (exclusive).
    window_size : int
        The size of the window used to calculate nearest neighbours.
    score : callable
        A callable that computes the score of a segmentation given the true and predicted labels.
        The callable must accept two arguments: y_true and y_pred, which are arrays of binary labels
        indicating whether each timepoint belongs to the first or second segment of the segmentation.

    Returns
    -------
    np.ndarray
        An array of shape (end-start,) containing the classification score profile.
    """
    profile = np.full(shape=end - start, fill_value=-np.inf, dtype=np.float64)

    for split_idx in range(start, end):
        y_true, y_pred = cross_val_labels(offsets, split_idx, window_size)
        profile[split_idx - start] = score(y_true, y_pred)

    return profile


@njit(fastmath=True, cache=True, parallel=True)
def _parallel_profile(offsets, pranges, window_size, score):
    """
    Computes the classification score profile given nearest neighbour offsets in parallel
    with n_jobs threads.

    Parameters
    ----------
    offsets : np.ndarray
        An array of shape (n_timepoints, k_neighbors) containing the offsets of the k nearest
        neighbors for each timepoint.
    pranges : ndarray of shape (m, 2), where each row is (start, end)
        Ranges in which the profile scpres are calculated per thread. Infers the number of threads.
    window_size : int
        The size of the window used to calculate nearest neighbours.
    score : callable
        A callable that computes the score of a segmentation given the true and predicted labels.
        The callable must accept two arguments: y_true and y_pred, which are arrays of binary labels
        indicating whether each timepoint belongs to the first or second segment of the segmentation.

    Returns
    -------
    np.ndarray
        An array of shape (n_timepoints,) containing the classification score profile.
    """
    profile = np.full(shape=offsets.shape[0], fill_value=-np.inf, dtype=np.float64)

    for idx in prange(len(pranges)):
        start, end = pranges[idx]
        profile[start:end] = _profile(offsets, start, end, window_size, score)

    return profile


class ClaSP:
    """
    An implementation of the ClaSP algorithm for detecting change points in time series data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window used for computing distances and offsets, by default 10.
    k_neighbours : int, optional
        The number of nearest neighbors to consider when computing distances and offsets, by default 3.
    distance: str
        The name of the distance function to be computed for determining the k-NNs. Available options are
        "znormed_euclidean_distance" and "euclidean_distance".
    score : str or callable, optional
        The name of the classification score to use.
        Available options are "roc_auc", "f1", by default "roc_auc".
    excl_radius : int, optional
        The radius of the exclusion zone around the detected change point, by default 5*window_size.
    n_jobs : int, optional (default=1)
        Amount of threads used in the ClaSP computation.

    Methods
    -------
    fit(time_series)
        Create a ClaSP for the input time series data.
    predict()
        Return the ClaSP for the input time series data.
    fit_predict(time_series)
        Create and return a ClaSP for the input time series data.
    split()
        Split ClaSP into two segments.
    """

    def __init__(self, window_size=10, score="roc_auc",
                 excl_radius=5, n_jobs=-1, random_state=2357):
        self.window_size = window_size
        self.score_name = score
        self.score = map_scores(score)
        self.excl_radius = excl_radius
        self.n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        self.random_state = random_state
        self.is_fitted = False

    def _check_is_fitted(self):
        """
        Checks if the ClaSP object is fitted.

        Raises
        ------
        NotFittedError
            If the ClaSP object is not fitted.

        Returns
        -------
        None
        """
        if not self.is_fitted:
            raise NotFittedError("ClaSP object is not fitted yet. Please fit the object before using this method.")

    def fit(self, time_series):
        """
        Fits the ClaSP model to the input time series data.

        Parameters
        ----------
        time_series : numpy.ndarray
            The input time series data to fit the model on.

        Returns
        -------
        self : ClaSP
            The fitted ClaSP object.

        Raises
        ------
        ValueError
            If the input time series has less than 2*min_seg_size data points.
        """
        check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius
        self.lbound, self.ubound = 0, time_series.shape[0]

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series

        self.offsets = SubsequenceRegressor(
            window_size=self.window_size,
            random_state=self.random_state
        ).fit(time_series).y_pred

        pranges = List()
        n_jobs = self.n_jobs

        while self.offsets.shape[0] // n_jobs < self.min_seg_size and n_jobs != 1:
            n_jobs -= 1

        bin_size = self.offsets.shape[0] // n_jobs

        for idx in range(n_jobs):
            start = max(idx * bin_size, self.min_seg_size)
            end = min((idx + 1) * bin_size, self.offsets.shape[0] - self.min_seg_size + self.window_size)
            if end > start: pranges.append((start, end))

        self.profile = _parallel_profile(self.offsets, pranges, self.window_size, self.score)

        self.is_fitted = True
        return self

    def transform(self):
        """
        Transform the input time series into a ClaSP profile.

        Returns
        -------
        profile : numpy.ndarray
            The ClaSP profile for the input time series.
        """
        self._check_is_fitted()
        return self.profile

    def fit_transform(self, time_series):
        """
        Fit the ClaSP algorithm to the given time series and return the
        corresponding profile.

        Parameters
        ----------
        time_series : np.ndarray, shape (n_timepoints,)
            The input time series to be segmented.

        Returns
        -------
        np.ndarray, shape (n_timepoints,)
            The ClaSP scores corresponding to each time point of the input time series.

        """
        return self.fit(time_series).transform()

    def split(self, sparse=True, validation="significance_test", threshold=1e-15):
        """
        Split the input time series into two segments using the change point location.

        Parameters
        ----------
        sparse : bool, optional
            If True, returns only the index of the change point. If False, returns the two segments
            separated by the change point. Default is True.
        validation : str, optional
            The validation method to use for determining the significance of the change point.
            The available methods are "significance_test" and "score_threshold". Default is
            "significance_test".
        threshold : float, optional
            The threshold value to use for the validation test. If the validation method is
            "significance_test", this value represents the p-value threshold for rejecting the
            null hypothesis. If the validation method is "score_threshold", this value represents
            the threshold score for accepting the change point. Default is 1e-15.

        Returns
        -------
        int or tuple
            If `sparse` is True, returns the index of the change point. If False, returns a tuple
            of the two time series segments separated by the change point.

        Raises
        ------
        ValueError
            If the `validation` parameter is not one of the available methods.
        """
        self._check_is_fitted()
        cp = np.argmax(self.profile)

        if validation is not None:
            validation_test = map_validation_tests(validation)
            if not validation_test(self, cp, threshold): return None

        if sparse is True:
            return cp

        return self.time_series[:cp], self.time_series[cp:]
