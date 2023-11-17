import logging
import sys

from aeon.annotation.ggs import GreedyGaussianSegmentation

from benchmark.clustering_test import evaluate_kshape, evaluate_kmeans, evaluate_gak, evaluate_time2feat, \
    evaluate_agglomerative, evaluate_spectral
from src.clap import CLaP
from external.competitor import autoplait

sys.path.insert(0, "../")

import pandas as pd

sys.path.insert(0, "../")

from benchmark.metrics import covering, f_measure
from src.utils import create_state_labels, load_has_datasets, extract_cps

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import numpy as np

np.random.seed(1379)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # dataset, w, cps, labels, ts = load_tssb_datasets(names=("Crop",)).iloc[0, :]  # Ham, MelbournePedestrian, Crop
    dataset, w, cps, labels, ts = load_has_datasets().iloc[0, :]  # Ham, MelbournePedestrian, Crop

    # load segmentation for ClaP
    segmentation_algorithm = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/TSSB_{segmentation_algorithm}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]
    # found_cps = seg_df.loc[seg_df["dataset"] == dataset].iloc[0].found_cps

    # clap = CLaP(n_jobs=4).fit(ts, found_cps)

    # found_cps = np.arange(pred_seg_labels.shape[0] - 1)[pred_seg_labels[:-1] != pred_seg_labels[1:]] + 1

    # found_cps = clap.get_change_points()
    # found_labels = clap.get_segment_labels()

    _, true_seg_labels, pred_seg_labels, ami = evaluate_spectral(dataset, w, cps, labels, ts)
    found_cps = extract_cps(pred_seg_labels)

    print(ts.shape[0])
    print(pred_seg_labels.shape[0])

    # true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    # pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)
    ami = np.round(adjusted_mutual_info_score(true_seg_labels, pred_seg_labels), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, AMI: {ami}") # , Labels: {found_labels}

