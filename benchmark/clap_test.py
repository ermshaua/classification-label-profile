import logging
import sys
sys.path.insert(0, "../")

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from benchmark.metrics import covering
from claspy.segmentation import BinaryClaSPSegmentation
from src.clap import CLaP
from src.utils import load_tssb_datasets, create_state_labels

import numpy as np

np.random.seed(1379)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = list()

    for _, (dataset, w, cps, labels, ts) in load_tssb_datasets().iterrows():
        clasp = BinaryClaSPSegmentation() # distance="combined"
        found_cps = clasp.fit_predict(ts)

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

        _, pred_labels = clap.get_change_points(), clap.get_segment_labels()

        true_seg_labels = create_state_labels(cps, labels, ts.shape[0])
        pred_seg_labels = create_state_labels(found_cps, pred_labels, ts.shape[0]) #

        ars = np.round(adjusted_rand_score(true_seg_labels, pred_seg_labels), 3)
        covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

        df.append((dataset, covering_score, ars, labels, pred_labels))
        print(f"{dataset}: Covering: {covering_score} ARS: {ars} Labels: {labels.tolist()} Predictions: {pred_labels.tolist()}")

    df = pd.DataFrame.from_records(df, columns=["dataset", "Covering", "ARS", "true_labels", "pred_labels"])
    print(f"Mean: {np.round(df.Covering.mean(), 3)}/{np.round(df.ARS.mean(), 3)} Std: {np.round(df.Covering.std(), 3)}/{np.round(df.ARS.std(), 3)}")
    df.to_csv("../tmp/clap.csv", index=False)
