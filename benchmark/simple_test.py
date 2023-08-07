import logging
import sys

import matplotlib.pyplot as plt

from src.convergent_neighbour import KConvergentSubsequenceNeighbours

sys.path.insert(0, "../")

from benchmark.metrics import covering
from claspy.segmentation import BinaryClaSPSegmentation
from src.clap import CLaP
from src.utils import load_tssb_datasets, create_state_labels, load_mosad_datasets, load_datasets

from sklearn.metrics import adjusted_rand_score, f1_score

import numpy as np

np.random.seed(1379)

DATASETS = [
    "Car",
    "ChlorineConcentration",
    "CinCECGTorso",
    "DistalPhalanxTW",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "FiftyWords",
    "Fish",
    "Haptics",
    "InlineSkate",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "MedicalImages",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "OSULeaf",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "SonyAIBORobotSurface1",
    "Symbols",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "c"
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dataset, w, cps, labels, ts = load_tssb_datasets(names=("Adiac",)).iloc[0,:]
    # dataset, w, cps, ts = load_datasets("TSSB", [0]).iloc[0,:]
    # dataset, routine, subject, sensor, sample_rate, cps, activities, ts = load_mosad_datasets().iloc[0,:]

    clasp = BinaryClaSPSegmentation(threshold=1e-6, n_jobs=1) # validation="custom", threshold=0.25 distance="combined"
    found_cps = clasp.fit_predict(ts)
    clasp.plot(gt_cps=cps, ts_name=dataset, file_path="../tmp/simple_test.pdf")

    exit(0)

    clap = CLaP(
        window_size=clasp.window_size,
        k_neighbours=clasp.k_neighbours,
        distance=clasp.distance,
    )

    if len(clasp.clasp_tree) > 0:
        knn = clasp.clasp_tree[0][1].knn
    else:
        knn = None

    clap.fit(ts, found_cps, knn=knn)

    found_cps = clap.get_change_points()
    pred_labels = clap.get_segment_labels()

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, pred_labels, ts.shape[0])

    ars = np.round(adjusted_rand_score(true_seg_labels, pred_seg_labels), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    print(f"{dataset}: Covering: {covering_score} ARS: {ars} Labels: {labels.tolist()} Predictions: {pred_labels.tolist()}")
    print(f"Score: {clap.score()}")
