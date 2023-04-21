import logging
import sys

from claspy.segmentation import BinaryClaSPSegmentation
from src.clap import CLaP
from src.utils import load_tssb_datasets, create_segmentation_labels

from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, "../")

import numpy as np

np.random.seed(1379)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dataset, w, cps, labels, ts = load_tssb_datasets(names=("CricketX",)).iloc[0,:]

    clasp = BinaryClaSPSegmentation(n_segments=len(cps)+1, validation=None)
    found_cps = clasp.fit_predict(ts)

    clap = CLaP(
        window_size=clasp.window_size,
        k_neighbours=clasp.k_neighbours,
        distance=clasp.distance,
        score=clasp.score
    )
    pred_labels = clap.fit(ts, found_cps).labels

    true_seg_labels = create_segmentation_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_segmentation_labels(found_cps, pred_labels, ts.shape[0])

    ars = np.round(adjusted_rand_score(true_seg_labels, pred_seg_labels), 3)

    print(f"ARS: {ars} Labels: {labels} Predictions: {pred_labels}")
