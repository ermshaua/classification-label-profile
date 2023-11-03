import os
import shutil
import sys

import stumpy
from aeon.annotation.ggs import GreedyGaussianSegmentation
from bocd import BayesianOnlineChangePointDetection, ConstantHazard, StudentT
from ruptures import Binseg, Window, Pelt
from stumpy import stump, fluss

sys.path.insert(0, "../")

import daproli as dp
import pandas as pd
from tqdm import tqdm

from benchmark.metrics import f_measure, covering
from claspy.segmentation import BinaryClaSPSegmentation
from src.utils import load_datasets, load_tssb_datasets


import numpy as np
np.random.seed(1379)


def evaluate_clasp(dataset, w, cps, labels, ts, **seg_kwargs):
    clasp = BinaryClaSPSegmentation()
    found_cps = clasp.fit_predict(ts)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps, clasp.profile)


def evaluate_binseg(dataset, w, cps, labels, ts, **seg_kwargs):
    # normalize ts to have comparable penalty influence
    ts = (ts - ts.min()) / (ts.max() - ts.min())

    binseg = Binseg(model="ar", min_size=int(ts.shape[0] * 0.05)).fit(ts)
    found_cps = np.array(binseg.predict(pen=0.2)[:-1], dtype=np.int64)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evaluate_window(dataset, w, cps, labels, ts, **seg_kwargs):
    # normalize ts to have comparable penalty influence
    ts = (ts - ts.min()) / (ts.max() - ts.min())

    binseg = Window(width=10*w, model="ar", min_size=int(ts.shape[0] * 0.05)).fit(ts)
    found_cps = np.array(binseg.predict(pen=0.2)[:-1], dtype=np.int64)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evaluate_pelt(dataset, w, cps, labels, ts, **seg_kwargs):
    # normalize ts to have comparable penalty influence
    ts = (ts - ts.min()) / (ts.max() - ts.min())

    binseg = Pelt(model="ar", min_size=int(ts.shape[0] * 0.05)).fit(ts)
    found_cps = np.array(binseg.predict(pen=0.2)[:-1], dtype=np.int64)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evaluate_fluss(dataset, w, cps, labels, ts, **seg_kwargs):
    mp = stump(ts, m=w)

    cac, regime_locations = fluss(mp[:, 1], L=w, n_regimes=len(cps) + 1)
    found_cps = regime_locations[cac[regime_locations] <= .45]

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps, cac)


def evaluate_bocd(dataset, w, cps, labels, ts, **seg_kwargs):
    # needs normalizing to not run into FloatingPointError
    ts = (ts - ts.min()) / (ts.max() - ts.min())

    bc = BayesianOnlineChangePointDetection(ConstantHazard(100), StudentT())
    profile = np.empty(ts.shape)

    for idx, timepoint in enumerate(ts):
        bc.update(timepoint)
        profile[idx] = bc.rt

    diff = np.diff(profile)
    found_cps = np.where(diff <= -250)[0]

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps, diff)


def evalute_segmentation_algorithm(dataset, n_timestamps, cps_true, cps_pred, profile=None):
    f1_score = np.round(f_measure({0: cps_true}, cps_pred, margin=int(n_timestamps * .01)), 3)
    covering_score = np.round(covering({0: cps_true}, cps_pred, n_timestamps), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering-Score: {covering_score}")

    if profile is not None:
        return dataset, cps_true.tolist(), cps_pred.tolist(), f1_score, covering_score, profile.tolist()

    return dataset, cps_true.tolist(), cps_pred.tolist(), f1_score, covering_score


def evaluate_candidate(dataset_name, candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    if dataset_name == "TSSB":
        df = load_tssb_datasets()  # names=REOCCURING_SEGMENTS
    else:
        df = load_datasets(dataset_name)

    df_cand = dp.map(
        lambda _, args: eval_func(*args, **seg_kwargs),
        tqdm(list(df.iterrows()), disable=verbose < 1),
        ret_type=list,
        verbose=0,
        n_jobs=n_jobs,
    )

    if columns is None:
        columns = ["dataset", "true_cps", "found_cps", "f1_score", "covering_score"]

    df_cand = pd.DataFrame.from_records(
        df_cand,
        index="dataset",
        columns=columns,
    )

    print(
        f"{candidate_name}: mean_f1_score={np.round(df_cand.f1_score.mean(), 3)}, mean_covering_score={np.round(df_cand.covering_score.mean(), 3)}")
    return df_cand


def evaluate_competitor(dataset_name, exp_path, n_jobs, verbose):
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)

    os.mkdir(exp_path)

    competitors = [
        ("ClaSP", evaluate_clasp),
        # ("BinSeg", evaluate_binseg),
        # ("Window", evaluate_window),
        # ("Pelt", evaluate_pelt),
        # ("FLUSS", evaluate_fluss),
        # ("BOCD", evaluate_bocd)
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        columns = None

        if candidate_name in ("ClaSP", "FLUSS", "BOCD"):
            columns = ["dataset", "true_cps", "found_cps", "f1_score", "covering_score", "profile"]

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=eval_func,
            columns=columns,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/segmentation/"
    n_jobs, verbose = 50, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor("TSSB", exp_path, n_jobs, verbose)
