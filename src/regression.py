import numpy as np
import pandas as pd

from aeon.transformations.collection.rocket import Rocket

import numpy as np
from numba import njit
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from claspy.utils import check_input_time_series
from src.contractable_neighbour import SubsequenceReferenceTransform
from src.utils import create_sliding_window, AeonTransformerWrapper


@njit(fastmath=True, cache=True)
def cross_val_labels(offsets, split_idx, window_size):
    """
    Generate predicted and true labels for cross-validation based on nearest neighbour distances.

    Parameters
    ----------
    offsets : ndarray of shape (n_timepoints, k_neighbours)
        The indices of the nearest neighbours for each timepoint in the time series. These indices
        are relative to the start of the time series and should be positive integers.
    split_idx : int
        The index at which to split the time series into two potential segments. This index should be
        less than n_timepoints and greater than window_size.
    window_size : int
        The size of the window used to calculate nearest neighbours.

    Returns
    -------
    y_true : ndarray of shape (n_timepoints,)
        The true labels for each timepoint in the time series.
    y_pred : ndarray of shape (n_timepoints,)
        The predicted labels for each timepoint in the time series.
    """
    n_timepoints = offsets.shape[0]

    y_true = np.concatenate((
        np.zeros(split_idx, dtype=np.int64),
        np.ones(n_timepoints - split_idx, dtype=np.int64),
    ))

    y_pred = np.asarray(offsets > split_idx, dtype=np.int64)

    exclusion_zone = np.arange(split_idx - window_size, split_idx)
    y_pred[exclusion_zone] = 1

    return y_true, y_pred


class SubsequenceRegressor:

    def __init__(self, window_size=10, random_state=2357):
        self.window_size = window_size
        self.random_state = random_state

    def fit(self, time_series):
        np.random.seed(self.random_state)
        check_input_time_series(time_series)

        if time_series.shape[0] < self.window_size:
            raise ValueError("Time series must at least have window_size data points.")

        self.time_series = time_series
        windows = create_sliding_window(self.time_series, self.window_size)
        windows = zscore(windows, axis=1)

        reference_transform = SubsequenceReferenceTransform(
            window_size= 50,# self.window_size,
            n_references=min(100, time_series.shape[0] - self.window_size + 1)
        ).fit(time_series)

        X_train = reference_transform.X
        X_train = StandardScaler().fit_transform(X_train)

        train_ind = np.arange(0, X_train.shape[0])
        # train_ind = np.random.choice(X_train.shape[0], int(X_train.shape[0] * .5), replace=True)

        # noise = np.random.choice(np.arange(-2*self.window_size, 2*self.window_size+1), X_train.shape[0], replace=True)
        noise = np.random.randn(X_train.shape[0]) * self.window_size
        y_train = (np.arange(X_train.shape[0])) / X_train.shape[0] # + noise

        self.regressor = Ridge()
        self.regressor.fit(X_train[train_ind,:], y_train[train_ind].reshape(-1,1)) #

        self.y_pred = np.asarray(self.regressor.predict(X_train).flatten() * X_train.shape[0], dtype=np.int64)

        return self