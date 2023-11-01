import sys
sys.path.insert(0, "../")

import os
import shutil

from src.competitor.hdp_hsmm import HDP_HSMM

from src.competitor.autoplait import autoplait

from src.competitor.ticc import TICC

from sklearn.metrics import adjusted_mutual_info_score

from src.competitor.time2state import CausalConv_LSE_Adaper, DPGMM, params_LSE, Time2State, normalize

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
    ami = np.round(adjusted_mutual_info_score(true_seg_labels, pred_seg_labels), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, AMI: {ami}")
    return dataset, cps.tolist(), found_cps.tolist(), pred_seg_labels, f1_score, covering_score, ami


def evaluate_time2state(dataset, w, cps, labels, ts, **seg_kwargs):
    window_size, step = 256, 10
    params_LSE['in_channels'] = 1
    params_LSE['out_channels'] = 2
    # params_LSE['compared_length'] = window_size

    data = normalize(np.array([ts]).reshape(-1, 1))
    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])

    try:
        t2s = Time2State(
            window_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)
        ).fit(data, window_size, step)

        pred_seg_labels = t2s.state_seq
    except ValueError as e:
        print(f"Exception: {e}; using only zero class.")
        pred_seg_labels = np.zeros_like(true_seg_labels)

    found_cps = np.arange(pred_seg_labels.shape[0] - 1)[pred_seg_labels[:-1] != pred_seg_labels[1:]] + 1

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)
    ami = np.round(adjusted_mutual_info_score(true_seg_labels, pred_seg_labels), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, AMI: {ami}")
    return dataset, cps.tolist(), found_cps.tolist(), pred_seg_labels, f1_score, covering_score, ami


def evaluate_ticc(dataset, w, cps, labels, ts, **seg_kwargs):
    num_state = np.unique(labels).shape[0]
    lambda_parameter = 1e-3
    beta = 2200
    threshold = 1e-4

    true_seg_labels = create_state_labels(cps, labels, ts.shape[0] - w + 1)
    data = np.array([ts]).reshape(-1, 1)

    try:
        ticc = TICC(window_size=w, number_of_clusters=num_state, lambda_parameter=lambda_parameter, beta=beta,
                    threshold=threshold, maxIters=10)

        pred_seg_labels, _ = ticc.fit_transform(data)
        pred_seg_labels = pred_seg_labels.astype(np.int64)
    except Exception as e:
        print(f"Exception: {e}; using only zero class.")
        pred_seg_labels = np.zeros_like(true_seg_labels)

    found_cps = np.arange(pred_seg_labels.shape[0] - 1)[pred_seg_labels[:-1] != pred_seg_labels[1:]] + 1

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)
    ami = np.round(adjusted_mutual_info_score(true_seg_labels, pred_seg_labels), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, AMI: {ami}")
    return dataset, cps.tolist(), found_cps.tolist(), pred_seg_labels, f1_score, covering_score, ami


def evaluate_autoplait(dataset, w, cps, labels, ts, **seg_kwargs):
    found_cps, found_labels = autoplait(dataset, ts, cps.shape[0])

    pred_seg_labels = create_state_labels(found_cps, found_labels, ts.shape[0])
    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)
    ami = np.round(adjusted_mutual_info_score(true_seg_labels, pred_seg_labels), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, AMI: {ami}")
    return dataset, cps.tolist(), found_cps.tolist(), pred_seg_labels, f1_score, covering_score, ami


def evaluate_hdp_hsmm(dataset, w, cps, labels, ts, **seg_kwargs):
    data = np.array([ts]).reshape(-1, 1)
    true_seg_labels = create_state_labels(cps, labels, ts.shape[0])

    try:
        hdp = HDP_HSMM(alpha=1e4, beta=20, n_iter=20)
        pred_seg_labels = hdp.fit_transform(data)
    except Exception as e:
        print(f"Exception: {e}; using only zero class.")
        pred_seg_labels = np.zeros_like(true_seg_labels)

    found_cps = np.arange(pred_seg_labels.shape[0] - 1)[pred_seg_labels[:-1] != pred_seg_labels[1:]] + 1

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)
    ami = np.round(adjusted_mutual_info_score(true_seg_labels, pred_seg_labels), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering: {covering_score}, AMI: {ami}")
    return dataset, cps.tolist(), found_cps.tolist(), pred_seg_labels, f1_score, covering_score, ami


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
        columns = ["dataset", "true_cps", "found_cps", "found_labels", "f1_score", "covering_score", "ami"]

    df_cand = pd.DataFrame.from_records(
        df_cand,
        index="dataset",
        columns=columns,
    )

    print(
        f"{candidate_name}: mean_f1_score={np.round(df_cand.f1_score.mean(), 3)}, mean_covering_score={np.round(df_cand.covering_score.mean(), 3)}, mean_ami_score={np.round(df_cand.ami.mean(), 3)}")
    return df_cand


def evaluate_competitor(dataset_name, exp_path, n_jobs, verbose):
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)

    os.mkdir(exp_path)

    competitors = [
        ("CLaP", evaluate_clap),
        # ("Time2State", evaluate_time2state),
        # ("TICC", evaluate_ticc)
        # ("AutoPlait", evaluate_autoplait)
        # ("HDP-HSMM", evaluate_hdp_hsmm)
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

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=eval_func,
            columns=None,
            n_jobs=n_jobs,
            verbose=verbose,
            segmentation=seg_df
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/state_detection/"
    n_jobs, verbose = 50, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor("TSSB", exp_path, n_jobs, verbose)
