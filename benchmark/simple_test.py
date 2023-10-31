import logging
import sys

from src.competitor.time2feat import feature_extraction, feature_selection, ClusterWrapper

sys.path.insert(0, "../")

import pandas as pd

sys.path.insert(0, "../")

from benchmark.metrics import covering, f_measure
from src.utils import load_tssb_datasets, create_state_labels, create_sliding_window, expand_label_sequence

from sklearn.metrics import adjusted_rand_score

import numpy as np

np.random.seed(1379)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dataset, w, cps, labels, ts = load_tssb_datasets(names=("ArrowHead",)).iloc[0, :]  # Ham, MelbournePedestrian, Crop

    # load segmentation for ClaP
    segmentation_algorithm = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/TSSB_{segmentation_algorithm}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]

    found_cps = seg_df.loc[seg_df["dataset"] == dataset].iloc[0].found_cps

    # clap = CLaP()
    # clap.fit(ts, found_cps)

    # as in CLaP
    sample_size = 2 * w
    stride = sample_size // 2

    windows = create_sliding_window(ts, sample_size, stride)[:10]
    # Time2Feat expects multivariate time series
    windows = np.array([np.array([w]) for w in windows])

    # model params
    transform_type = 'minmax'
    model_type = 'Hierarchical'
    context = {'model_type': model_type, 'transform_type': transform_type}

    try:
        df_features = feature_extraction(windows, batch_size=500)
        top_features = feature_selection(df_features, None, context)
        df_features = df_features[top_features]
        model = ClusterWrapper(n_clusters=np.unique(labels).shape[0], model_type=model_type,
                               transform_type=transform_type)
        pred = model.fit_predict(df_features.values)
    except Exception as e:
        print(f"Exception: {e}; using only zero class.")
        pred = np.zeros(shape=windows.shape[0], dtype=np.int64)

    # found_cps = clap.get_change_points()
    # found_labels = clap.get_segment_labels()

    pred_seg_labels = expand_label_sequence(pred, sample_size, stride)
    true_seg_labels = create_state_labels(cps, labels, pred_seg_labels.shape[0])
    # pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)
    ars = np.round(adjusted_rand_score(true_seg_labels, pred_seg_labels), 3)

    print(
        f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, ARS: {ars}, Labels: {labels}")  # , Predictions: {found_labels}
    # print(f"Score: {clap.score()}")
