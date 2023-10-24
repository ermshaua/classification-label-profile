import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.contractable_neighbour import KContractableSubsequenceNeighbours

sys.path.insert(0, "../")

from benchmark.metrics import covering, f_measure
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
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dataset, w, cps, labels, ts = load_tssb_datasets(names=("MelbournePedestrian",)).iloc[0,:] # Ham, MelbournePedestrian, Crop

    # load segmentation for ClaP
    segmentation_algorithm = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/TSSB_{segmentation_algorithm}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]

    found_cps = seg_df.loc[seg_df["dataset"] == dataset].iloc[0].found_cps

    clap = CLaP()
    clap.fit(ts, found_cps)

    found_cps = clap.get_change_points()
    found_labels = clap.get_segment_labels()

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)
    ars = np.round(adjusted_rand_score(true_seg_labels, pred_seg_labels), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, ARS: {ars}, Labels: {labels}, Predictions: {found_labels}")
    print(f"Score: {clap.score()}")
