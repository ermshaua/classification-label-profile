import os
import shutil
import sys
from itertools import product

sys.path.insert(0, "../")

from src.clap import CLaP

import daproli as dp
import pandas as pd
from tqdm import tqdm

from benchmark.metrics import f_measure, covering
from claspy.segmentation import BinaryClaSPSegmentation
from src.utils import load_datasets

import numpy as np

np.random.seed(1379)


def cross_val_clasp_clap(ts):
    scores = []
    clasps, claps = [], []
    window_sizes = ["suss"]  # , "acf", "fft", 10, 20, 50, 100
    # distances = ["euclidean_distance", "znormed_euclidean_distance"]

    for window_size in window_sizes:
        clasp = BinaryClaSPSegmentation(window_size=window_size)  # , validation="custom", threshold=0.15

        try:
            found_cps = clasp.fit_predict(ts)
        except:
            continue

        clap = CLaP(
            window_size=clasp.window_size,
            k_neighbours=clasp.k_neighbours,
            distance=clasp.distance,
        )

        if len(clasp.clasp_tree) > 0:
            knn = clasp.clasp_tree[0][1].knn
        else:
            knn = None

        try:
            clap.fit(ts, found_cps, knn=knn)
        except:
            continue

        clasps.append(clasp)
        claps.append(clap)
        scores.append(clap.score())

    best_idx = np.argmax(scores)
    return clasps[best_idx], claps[best_idx], claps[best_idx].get_change_points()


def cross_val_clasp(ts):
    scores = []
    clasps = []

    window_sizes = ["acf", "fft", "suss"] # np.arange(5, 100 + 1, 5)
    distances = ["euclidean_distance", "znormed_euclidean_distance"]
    thresholds = (1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30)
    min_seg_sizes = (5,10,20,50)

    for w, d in product(window_sizes, distances):
        clasp = BinaryClaSPSegmentation(window_size=w, distance=d)

        try:
            found_cps = clasp.fit_predict(ts)
        except:
            continue

        clasps.append(clasp)

        complexity = np.sqrt(np.sum(np.square(np.diff(clasp.profile))))
        score = complexity * (np.max(clasp.profile) if len(clasp.scores) == 0 else np.max(clasp.scores))

        scores.append(score)

    args = np.argsort(scores)

    scores = np.asarray(scores)[args]
    scores = np.abs(np.diff(scores))

    clasps = np.asarray(clasps)[args]

    best_idx = np.argmax(scores) + 1
    # best_idx = np.argmax(scores)

    return clasps[best_idx], clasps[best_idx].predict()


def evaluate_clasp(dataset, w, cps, ts, **seg_kwargs):
    # clasp, found_cps = cross_val_clasp(ts)

    clasp = BinaryClaSPSegmentation(threshold=1e-6, n_jobs=-1)
    found_cps = clasp.fit_predict(ts)

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    print(f"{dataset}: F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return dataset, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score,
                                                                                      3), clasp.profile.tolist()


def evaluate_candidate(candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    df = load_datasets("TSSB")  #

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


def evaluate_competitor(exp_path, n_jobs, verbose):
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)

    os.mkdir(exp_path)

    competitors = [
        ("ClaSPy", evaluate_clasp),
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        columns = None

        if candidate_name in ("ClaSPy", "FLUSS", "FLOSS"):
            columns = ["dataset", "true_cps", "found_cps", "f1_score", "covering_score", "profile"]

        df = evaluate_candidate(
            candidate_name,
            eval_func=eval_func,
            columns=columns,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = 20, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor(exp_path, n_jobs, verbose)
