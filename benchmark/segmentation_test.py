import os
import shutil
import sys
import warnings
from ctypes import Union

import stumpy
from aeon.annotation.ggs import GreedyGaussianSegmentation
from bocd import BayesianOnlineChangePointDetection, ConstantHazard, StudentT
from roerich.algorithms import ChangePointDetectionClassifier, ChangePointDetectionRuLSIF
from ruptures import Binseg, Window, Pelt
from sklearn.cluster import AgglomerativeClustering
from stumpy import stump, fluss
from stumpy.floss import _cac

sys.path.insert(0, "../")

import daproli as dp
import pandas as pd
from tqdm import tqdm

from benchmark.metrics import f_measure, covering
from claspy.segmentation import BinaryClaSPSegmentation
from src.utils import load_datasets, load_tssb_datasets, load_has_datasets

import numpy as np
np.random.seed(1379)


def evaluate_clasp(dataset, w, cps, labels, ts, **seg_kwargs):
    if ts.ndim == 1:
        clasp = BinaryClaSPSegmentation(n_jobs=1)
        found_cps = clasp.fit_predict(ts)
        profile = clasp.profile
    else:
        window_sizes, profiles, found_cps, scores, ind = [], [], [], [], []

        for dim in range(ts.shape[1]):
            clasp = BinaryClaSPSegmentation(n_jobs=1).fit(ts[:,dim])

            window_sizes.append(clasp.window_size)
            profiles.append(clasp.profile)
            scores.extend(clasp.scores) # todo: this should be fixed in claspy: np.array(clasp.change_points, dtype=int)
            found_cps.extend(clasp.change_points)
            ind.extend([len(window_sizes)-1] * len(clasp.change_points))

        min_len = np.min([len(p) for p in profiles])
        profile = np.max([p[:min_len] for p in profiles], axis=0)

        found_cps = np.array(found_cps)
        scores = np.array(scores)

        min_match = int(np.log2(len(profiles)))

        if len(found_cps) >= min_match:
            clu = AgglomerativeClustering(n_clusters=None, linkage="average", distance_threshold=5 * np.mean(window_sizes))
            clusters = clu.fit_predict(found_cps.reshape(-1, 1))

            merged_cps = []

            for label in np.unique(clusters):
                candidates = found_cps[clusters == label]
                candidate_scores = scores[clusters == label]

                if len(candidates) < min_match: continue
                merged_cps.append(int(np.mean(candidates)))

            found_cps = np.sort(merged_cps)
        else:
            found_cps = np.array([])

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evaluate_binseg(dataset, w, cps, labels, ts, **seg_kwargs):
    # normalize ts to have comparable penalty influence
    ts = (ts - ts.min()) / (ts.max() - ts.min())

    binseg = Binseg(model="ar", min_size=5 * w).fit(ts)
    found_cps = np.array(binseg.predict(pen=0.2)[:-1], dtype=np.int64)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evaluate_window(dataset, w, cps, labels, ts, **seg_kwargs):
    # normalize ts to have comparable penalty influence
    ts = (ts - ts.min()) / (ts.max() - ts.min())

    binseg = Window(width=10*w, model="ar", min_size=5 * w).fit(ts)
    found_cps = np.array(binseg.predict(pen=0.2)[:-1], dtype=np.int64)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evaluate_pelt(dataset, w, cps, labels, ts, **seg_kwargs):
    # normalize ts to have comparable penalty influence
    ts = (ts - ts.min()) / (ts.max() - ts.min())

    binseg = Pelt(model="ar", min_size=5 * w).fit(ts)
    found_cps = np.array(binseg.predict(pen=0.2)[:-1], dtype=np.int64)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evaluate_fluss(dataset, w, cps, labels, ts, **seg_kwargs):
    if ts.ndim == 1:
        mp = stump(ts, m=w)
        cac = _cac(mp[:, 1], L=w)
    else:
        C = []

        for dim in range(ts.shape[1]):
            mp = stump(ts[:,dim], m=w)
            C.append(_cac(mp[:, 1], L=w))

        cac = np.mean(C, axis=0)

    threshold = .45
    found_cps = []

    profile = np.copy(cac)

    while profile.min() <= threshold:
        cp = np.argmin(profile)
        found_cps.append(cp)
        profile[max(0, cp - 5 * w):min(profile.shape[0], cp + 5 * w)] = np.inf

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, np.array(found_cps))


def evaluate_ddre(dataset, w, cps, labels, ts, **seg_kwargs):
    cpdc = ChangePointDetectionClassifier(base_classifier='logreg', metric='klsym', periods=1, window_size=5 * w)

    try:
        # fails sometimes
        _, found_cps = cpdc.predict(ts)
    except ValueError:
        found_cps = np.empty(0, dtype=int)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evaluate_rulsif(dataset, w, cps, labels, ts, **seg_kwargs):
    rulsif = ChangePointDetectionRuLSIF(periods=1, window_size=int(5*w), step=5, n_runs=1)

    try:
        # fails sometimes
        _, found_cps = rulsif.predict(ts)
    except ValueError:
        found_cps = np.empty(0, dtype=int)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps, found_cps)


def evalute_segmentation_algorithm(dataset, n_timestamps, cps_true, cps_pred, profile=None):
    f1_score = np.round(f_measure({0: cps_true}, cps_pred, margin=int(n_timestamps * .01)), 3)
    covering_score = np.round(covering({0: cps_true}, cps_pred, n_timestamps), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering-Score: {covering_score}")

    if profile is not None:
        return dataset, cps_true.tolist(), cps_pred.tolist(), f1_score, covering_score, profile.tolist()

    return dataset, cps_true.tolist(), cps_pred.tolist(), f1_score, covering_score


def evaluate_candidate(dataset_name, candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    if dataset_name == "TSSB":
        df = load_tssb_datasets()
    elif dataset_name == "HAS":
        df = load_has_datasets()
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
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    competitors = [
        ("ClaSP", evaluate_clasp),
        ("BinSeg", evaluate_binseg),
        ("Window", evaluate_window),
        ("Pelt", evaluate_pelt),
        ("FLUSS", evaluate_fluss),
        ("DDRE", evaluate_ddre),
        ("RuLSIF", evaluate_rulsif)
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=eval_func,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/segmentation/"
    n_jobs, verbose = 50, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor("HAS", exp_path, n_jobs, verbose)
