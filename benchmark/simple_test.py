import logging
import sys

from aeon.annotation.ggs import GreedyGaussianSegmentation

from src.competitor.autoplait import autoplait
from src.competitor.hdp_hsmm import HDP_HSMM
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

    data = np.array([ts]).reshape(-1, 1)

    ggs = GreedyGaussianSegmentation(k_max=len(cps))
    pred_seg_labels = ggs.fit_predict(data)

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])

    found_cps = np.arange(pred_seg_labels.shape[0] - 1)[pred_seg_labels[:-1] != pred_seg_labels[1:]] + 1

    # found_cps = clap.get_change_points()
    # found_labels = clap.get_segment_labels()

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)
    ars = np.round(adjusted_rand_score(true_seg_labels, pred_seg_labels), 3)

    print(
        f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, ARS: {ars}, Labels: {labels}")  # , Predictions: {found_labels}
    # print(f"Score: {clap.score()}")
