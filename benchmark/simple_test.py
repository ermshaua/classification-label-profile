import logging
import sys
sys.path.insert(0, "../")

from benchmark.metrics import covering
from claspy.segmentation import BinaryClaSPSegmentation
from src.clap import CLaP
from src.utils import load_tssb_datasets, create_state_labels

from sklearn.metrics import adjusted_rand_score

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
    "WordSynonyms"
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dataset, w, cps, labels, ts = load_tssb_datasets(names=("ArrowHead",)).iloc[0,:]

    clasp = BinaryClaSPSegmentation()
    found_cps = clasp.fit_predict(ts)
    pred_labels = clasp.labels

    # found_cps = np.arange(0, ts.shape[0]-w, w * 5)

    clap = CLaP(window_size=w) #
    pred_labels = clap.fit(ts, found_cps).labels

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, pred_labels, ts.shape[0])

    ars = np.round(adjusted_rand_score(true_seg_labels, pred_seg_labels), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    print(f"{dataset}: Covering: {covering_score} ARS: {ars} Labels: {labels.tolist()} Predictions: {pred_labels.tolist()}")
    print(found_cps)
