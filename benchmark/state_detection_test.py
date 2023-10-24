import os
import shutil
import sys

from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, "../")

import daproli as dp
import pandas as pd
from tqdm import tqdm

from benchmark.metrics import f_measure, covering
from src.clap import CLaP
from src.utils import create_state_labels, load_tssb_datasets

import numpy as np

np.random.seed(1379)

REOCCURING_SEGMENTS = [
    "Crop",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "FreezerRegularTrain",
    "Ham",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "ProximalPhalanxOutlineCorrect",
    "Strawberry",
]


def evaluate_clap(dataset, w, cps, labels, ts, **seg_kwargs):
    seg_df = seg_kwargs["segmentation"]
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

    print(f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, ARS: {ars}")
    return dataset, cps.tolist(), found_cps.tolist(), found_labels, f1_score, covering_score, ars


def evaluate_candidate(dataset_name, candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    if dataset_name != "TSSB":
        raise ValueError("Only TSSB dataset implemented.")

    df = load_tssb_datasets() # names=REOCCURING_SEGMENTS

    df_cand = dp.map(
        lambda _, args: eval_func(*args, **seg_kwargs),
        tqdm(list(df.iterrows()), disable=verbose < 1),
        ret_type=list,
        verbose=0,
        n_jobs=n_jobs,
    )

    if columns is None:
        columns = ["dataset", "true_cps", "found_cps", "found_labels", "ars"]

    df_cand = pd.DataFrame.from_records(
        df_cand,
        index="dataset",
        columns=columns,
    )

    print(
        f"{candidate_name}: mean_f1_score={np.round(df_cand.f1_score.mean(), 3)}, mean_covering_score={np.round(df_cand.covering_score.mean(), 3)}, mean_ars_score={np.round(df_cand.ars.mean(), 3)}")
    return df_cand


def evaluate_competitor(dataset_name, exp_path, n_jobs, verbose):
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)

    os.mkdir(exp_path)

    competitors = [
        ("CLaP", evaluate_clap),
    ]

    # load segmentation for ClaP
    segmentation_algorithm = "ClaSP"
    converters = dict([(column, lambda data: np.array(eval(data))) for column in ["found_cps"]])
    seg_df = pd.read_csv(
        f"../experiments/segmentation/{dataset_name}_{segmentation_algorithm}.csv.gz",
        converters=converters
    )[["dataset", "found_cps"]]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        columns = None

        if candidate_name in ("CLaP"):
            columns = ["dataset", "true_cps", "found_cps", "found_labels", "f1_score", "covering_score", "ars"]

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=eval_func,
            columns=columns,
            n_jobs=n_jobs,
            verbose=verbose,
            segmentation=seg_df
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/state_detection/"
    n_jobs, verbose = 20, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor("TSSB", exp_path, n_jobs, verbose)
