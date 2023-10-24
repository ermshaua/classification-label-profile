import os
import shutil
import sys

sys.path.insert(0, "../")

import daproli as dp
import pandas as pd
from tqdm import tqdm

from benchmark.metrics import f_measure, covering
from claspy.segmentation import BinaryClaSPSegmentation
from src.utils import load_datasets

import numpy as np
np.random.seed(1379)


def evaluate_clasp(dataset, w, cps, ts, **seg_kwargs):
    clasp = BinaryClaSPSegmentation()
    found_cps = clasp.fit_predict(ts)

    f1_score = np.round(f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01)), 3)
    covering_score = np.round(covering({0: cps}, found_cps, ts.shape[0]), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering-Score: {covering_score}")
    return dataset, cps.tolist(), found_cps.tolist(), f1_score, covering_score, clasp.profile.tolist()


def evaluate_candidate(dataset_name, candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    df = load_datasets(dataset_name)  #

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
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        columns = None

        if candidate_name in ("ClaSP", "FLUSS", "FLOSS"):
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
    n_jobs, verbose = 20, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_competitor("TSSB", exp_path, n_jobs, verbose)
