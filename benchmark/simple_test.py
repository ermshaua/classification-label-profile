import logging
import sys

from aeon.annotation.ggs import GreedyGaussianSegmentation

from benchmark.clustering_test import evaluate_kshape, evaluate_kmeans, evaluate_gak, evaluate_time2feat, \
    evaluate_agglomerative, evaluate_spectral
from benchmark.state_detection_test import evaluate_clap, evaluate_state_detection_algorithm
from src.clap import CLaP
from external.competitor import autoplait

sys.path.insert(0, "../")

import pandas as pd

sys.path.insert(0, "../")

from benchmark.metrics import covering, f_measure
from src.utils import create_state_labels, load_has_datasets, extract_cps, load_tssb_datasets, load_datasets

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import numpy as np

np.random.seed(1379)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    benchmark = "TSSB"

    # Ham, MelbournePedestrian, Crop
    if benchmark == "TSSB":
        df_data = load_tssb_datasets()
    elif benchmark == "HAS":
        df_data = load_has_datasets()
    else:
        df_data = load_datasets(benchmark)

    idx = 15 # 15, 43
    dataset, w, cps, labels, ts = df_data.iloc[idx, :]

    # load segmentation for ClaP
    segmentation_algorithm = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/{benchmark}_{segmentation_algorithm}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]

    found_cps = seg_df.loc[seg_df["dataset"] == dataset].iloc[0].found_cps

    clap = CLaP(n_jobs=4)
    clap.fit(ts, found_cps)

    found_cps = clap.get_change_points()
    found_labels = clap.get_segment_labels()

    print(labels)
    print(found_labels)

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    evaluate_state_detection_algorithm(dataset, ts.shape[0], cps, found_cps, true_seg_labels, pred_seg_labels, verbose=1)

